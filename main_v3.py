import re
import random
import time
from statistics import mode

from PIL import Image
import numpy as np
import pandas
import torch
import torch.nn as nn
from torch.utils.data import Subset, DataLoader
import torchvision
from torchvision import transforms
import transformers
from transformers import(
    BertTokenizer,
    BertModel,
    AutoModel,
    AutoTokenizer
)
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def process_text(text):
    # lowercase
    text = text.lower()

    # 数詞を数字に変換
    num_word_to_digit = {
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
        'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
        'ten': '10'
    }
    for word, digit in num_word_to_digit.items():
        text = text.replace(word, digit)

    # 小数点のピリオドを削除
    text = re.sub(r'(?<!\d)\.(?!\d)', '', text)

    # 冠詞の削除
    text = re.sub(r'\b(a|an|the)\b', '', text)

    # 短縮形のカンマの追加
    contractions = {
        "dont": "don't", "isnt": "isn't", "arent": "aren't", "wont": "won't",
        "cant": "can't", "wouldnt": "wouldn't", "couldnt": "couldn't"
    }
    for contraction, correct in contractions.items():
        text = text.replace(contraction, correct)

    # 句読点をスペースに変換
    text = re.sub(r"[^\w\s':]", ' ', text)

    # 句読点をスペースに変換
    text = re.sub(r'\s+,', ',', text)

    # 連続するスペースを1つに変換
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# 1. データローダーの作成
class VQADataset(torch.utils.data.Dataset):
    def __init__(self, df_path, image_dir, transform=None, answer=True, zca=None):
        self.transform = transform  # 画像の前処理
        self.image_dir = image_dir  # 画像ファイルのディレクトリ
        self.df = pandas.read_json(df_path)  # 画像ファイルのパス，question, answerを持つDataFrame
        self.answer = answer
        # self.zca = zca

        # question / answerの辞書を作成
        self.question2idx = {}
        self.answer2idx = {}
        self.idx2question = {}
        self.idx2answer = {}

        # 回答一覧class_mappingを登録する
        # self.answer2classid_df = pandas.read_csv('data/class_mapping.csv')

        # 質問文に含まれる単語を辞書に追加
        for question in self.df["question"]:
            question = process_text(question)
            words = question.split(" ")
            for word in words:
                if word not in self.question2idx:
                    self.question2idx[word] = len(self.question2idx)
        self.idx2question = {v: k for k, v in self.question2idx.items()}  # 逆変換用の辞書(question)

        if self.answer:
            # ここに存在しないものが出てきたら追加すれば良い
            # self.idx2answer = pandas.read_csv('data/class_mapping.csv').to_dict()['answer']
            self.idx2answer = pandas.read_csv('data/class_mapping.csv')
            self.idx2answer = self.idx2answer[:1000]
            self.idx2answer = self.idx2answer.to_dict()['answer']
            self.answer2idx = {v: k for k, v in self.idx2answer.items()}

            # 回答に含まれる単語を辞書に追加
            # answersには10組の答えが格納されている
            for answers in self.df["answers"]:
                for answer in answers:
                    word = answer["answer"]
                    word = process_text(word)
                    if word not in self.answer2idx:
                        self.answer2idx[word] = len(self.answer2idx)
                        self.idx2answer[len(self.idx2answer)] = word
             


        # 辞書など使わずにBERTから分散表現を獲得
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        # self.tokenizer = AutoTokenizer.from_pretrained('princeton-nlp/sup-simcse-roberta-base')
        # self.model = AutoModel.from_pretrained('princeton-nlp/sup-simcse-roberta-base').to('cuda')
        self.model = AutoModel.from_pretrained('bert-base-uncased').to('cuda')
        # self.model.resize_token_embeddings(len(self.question2idx))
    #   768次元の分散表現を得る
    def get_embedding(self, text):
        input_ids = self.tokenizer(text, return_tensors='pt').to('cuda')
        outputs = self.model(**input_ids, output_hidden_states=True)
        # sentence_vector = outputs.pooler_output
        last_hidden_state = outputs.last_hidden_state
        # print(last_hidden_state.shape)
        # Mean Poolingにより一定のサイズの分散表現を得る
        sentence_vector = last_hidden_state.mean(dim=1).to('cpu')
        # print(sentence_vector.shape)
        return sentence_vector
    
    # def get_embedding(self, text):
    #     input_ids = self.tokenizer(text, return_tensors='pt')
    #     # print(input_ids)
    #     return self.model(input_ids["input_ids"])

    def update_dict(self, dataset):
        """
        検証用データ，テストデータの辞書を訓練データの辞書に更新する．

        Parameters
        ----------
        dataset : Dataset
            訓練データのDataset
        """
        self.question2idx = dataset.question2idx
        self.answer2idx = dataset.answer2idx
        self.idx2question = dataset.idx2question
        self.idx2answer = dataset.idx2answer

    def __getitem__(self, idx):
        """
        対応するidxのデータ（画像，質問，回答）を取得．

        Parameters
        ----------
        idx : int
            取得するデータのインデックス

        Returns
        -------
        image : torch.Tensor  (C, H, W)
            画像データ
        question : torch.Tensor  (vocab_size)
            質問文をone-hot表現に変換したもの -> 分散表現
        answers : torch.Tensor  (n_answer)
            10人の回答者の回答のid
        mode_answer_idx : torch.Tensor  (1)
            10人の回答者の回答の中で最頻値の回答のid
        """
        image = Image.open(f"{self.image_dir}/{self.df['image'][idx]}")
        # zca = ZCAWhitening()
        image = self.transform(image)
        # zca.fit(image)
        # question = np.zeros(len(self.idx2question) + 1)  # 未知語用の要素を追加
        # question_words = self.df["question"][idx].split(" ")
        # for word in question_words:
        #     try:
        #         question[self.question2idx[word]] = 1  # one-hot表現に変換
        #     except KeyError:
        #         question[-1] = 1  # 未知語
        question = self.get_embedding(self.df["question"][idx])

        def count_answer_id(answers:list):
            answer_dict = {}    # {id: num}
            confidence2weight = {"yes":1.0, "maybe":0.5, "no":0.1}
            for answer in answers:
                confidence = confidence2weight[answer['answer_confidence']]
                answer_id = self.answer2idx[process_text(answer['answer'])]
                if answer_id not in answer_dict:
                    answer_dict[answer_id] = confidence
                else:
                    answer_dict[answer_id] = answer_dict[answer_id] + confidence

            return answer_dict
        


        if self.answer:
            answers = [self.answer2idx[process_text(answer["answer"])] for answer in self.df["answers"][idx]]
            answer_dict = count_answer_id(self.df['answers'][idx])
            # mode_answer_idx = mode(answers)  # 最頻値を取得（正解ラベル）
            mode_answer_idx = max(answer_dict,key=answer_dict.get)

            return image, torch.Tensor(question), torch.Tensor(answers), int(mode_answer_idx)

        else:
            return image, torch.Tensor(question)

    def __len__(self):
        return len(self.df)


# 2. 評価指標の実装
# 簡単にするならBCEを利用する
def VQA_criterion(batch_pred: torch.Tensor, batch_answers: torch.Tensor):
    total_acc = 0.

    for pred, answers in zip(batch_pred, batch_answers):
        acc = 0.
        for i in range(len(answers)):
            num_match = 0
            for j in range(len(answers)):
                if i == j:
                    continue
                if pred == answers[j]:
                    num_match += 1
            acc += min(num_match / 3, 1)
        total_acc += acc / 10

    return total_acc / len(batch_pred)


# 3. モデルの実装
# ResNetを利用できるようにしておく
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out += self.shortcut(residual)
        out = self.relu(out)

        return out


class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        out += self.shortcut(residual)
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers):
        super().__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, layers[0], 64)
        self.layer2 = self._make_layer(block, layers[1], 128, stride=2)
        self.layer3 = self._make_layer(block, layers[2], 256, stride=2)
        self.layer4 = self._make_layer(block, layers[3], 512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, 512)

    def _make_layer(self, block, blocks, out_channels, stride=1):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
    


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet50():
    return ResNet(BottleneckBlock, [3, 4, 6, 3])


class VQAModel(nn.Module):
    def __init__(self, vocab_size: int, n_answer: int):
        super().__init__()
        # self.resnet = ResNet18()
        # self.resnet = VGG19()
        self.resnet = ResNet50()
        # self.text_encoder = nn.Linear(vocab_size, 512)
        print(n_answer)
        self.fc = nn.Sequential(
            nn.Linear(768+512, int(n_answer/5)),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(int(n_answer/5),n_answer)
            # nn.ReLU(inplace=True),
            # nn.Dropout(0.25),
            # nn.Linear(256, n_answer)
        )

    def forward(self, image, question):
        image_feature = self.resnet(image)  # 画像の特徴量(dim=512)
        question_feature = question[:,0,:]  # テキストの特徴量(dim=1024)[CSL]トークン

        # print(image_feature.shape, question_feature.shape)
        x = torch.cat([image_feature, question_feature], dim=1)
        x = self.fc(x)

        return x


# 4. 学習の実装
def train(model, train_dataloader, valid_dataloader, optimizer, criterion, device):
    model.train()
    
    total_loss = 0
    total_acc = 0
    simple_acc = 0

    from tqdm.auto import tqdm

    start = time.time()
    for image, question, answers, mode_answer in tqdm(train_dataloader):
        image, question, answer, mode_answer = \
            image.to(device), question.to(device), answers.to(device), mode_answer.to(device)

        # print(image.shape, question.shape)
        pred = model(image, question)
        # print(pred.shape)
        # print(mode_answer.squeeze().shape)
        loss = criterion(pred, mode_answer.squeeze())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += VQA_criterion(pred.argmax(1), answers)  # VQA accuracy
        simple_acc += (pred.argmax(1) == mode_answer).float().mean().item()  # simple accuracy

    avg_train_loss = total_loss / len(train_dataloader)
    avg_train_acc = total_acc / len(train_dataloader)
    avg_train_simple_acc = simple_acc / len(train_dataloader)
    train_time = time.time() - start

    model.eval()
    val_loss = 0
    val_acc = 0
    val_simple_acc = 0

    with torch.no_grad():
        for image, question, answers, mode_answer in valid_dataloader:
            image, question, answer, mode_answer = \
                image.to(device), question.to(device), answers.to(device), mode_answer.to(device)
            
            pred = model(image, question)
            # print(pred.shape)
            # print(mode_answer.squeeze())

            loss = criterion(pred, mode_answer.squeeze())

            val_loss += loss.item()
            val_acc += VQA_criterion(pred.argmax(1), answers)
            val_simple_acc += (pred.argmax(1) == mode_answer).float().mean().item()

        avg_val_loss = val_loss / len(valid_dataloader)
        avg_val_acc = val_acc / len(valid_dataloader)
        avg_val_simple_acc = val_simple_acc / len(valid_dataloader)

    return {
            'train_loss': avg_train_loss,
            'train_acc': avg_train_acc,
            'train_simple_acc': avg_train_simple_acc,
            'train_time': train_time,
            'valid_loss': avg_val_loss,
            'valid_acc': avg_val_acc,
            'valid_simple_acc': avg_val_simple_acc,
        }


def eval(model, dataloader, optimizer, criterion, device):
    model.eval()

    total_loss = 0
    total_acc = 0
    simple_acc = 0

    start = time.time()
    for image, question, answers, mode_answer in dataloader:
        image, question, answer, mode_answer = \
            image.to(device), question.to(device), answers.to(device), mode_answer.to(device)

        pred = model(image, question)
        # print(pred.shape)
        # print(mode_answer.squeeze())
        # print(mode_answer.squeeze().shape)
        loss = criterion(pred, mode_answer.squeeze())

        total_loss += loss.item()
        total_acc += VQA_criterion(pred.argmax(1), answers)  # VQA accuracy
        simple_acc += (pred.argmax(1) == mode_answer).mean().item()  # simple accuracy

    return total_loss / len(dataloader), total_acc / len(dataloader), simple_acc / len(dataloader), time.time() - start


def main():
    # deviceの設定
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # dataloader / model
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.RandomRotation(degrees=(-180,180)),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
        transforms.ToTensor(),
        # transforms.RandomErasing(p=0.2,scale=(0.02,0.33),ratio=(0.3,3.3)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])

    train_dataset = VQADataset(df_path="./data/train.json", image_dir="./data/train", transform=transform_train)
    test_dataset = VQADataset(df_path="./data/valid.json", image_dir="./data/valid", transform=transform_test, answer=False)
    test_dataset.update_dict(train_dataset)

    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    # optimizer / criterion
    num_epoch = 5

    # Cross Validation
    skf = KFold(n_splits=10, shuffle=True, random_state=42)
    best_model_state = None
    best_valid_loss = float('inf')

    for fold, (train_idx, valid_idx) in enumerate(skf.split(train_dataset)):
        print("#"*25 + f"Kfold:{fold+1}" + "#"*25)

        hist_train_loss = []
        hist_valid_loss = []
        hist_train_acc = []
        hist_valid_acc = []
        model = VQAModel(vocab_size=len(train_dataset.question2idx)+1, n_answer=len(train_dataset.answer2idx)).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        train_data = Subset(train_dataset,train_idx)
        valid_data = Subset(train_dataset,valid_idx)
        train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
        valid_loader = DataLoader(valid_data, batch_size=128, shuffle=False)
        # print(train_data[0].shape)
        # print(valid_data[0].shape)

        for epoch in range(num_epoch):
            output = train(model, train_loader, valid_loader, optimizer, criterion, device)
            hist_train_loss.append(output['train_loss'])
            hist_train_acc.append(output['train_acc'])
            hist_valid_loss.append(output['valid_loss'])
            hist_valid_acc.append(output['valid_acc'])

            print(f"【{epoch + 1}/{num_epoch}】\n"
                  f"train time: {output['train_time']:.2f} [s]\n"
                  f"train loss: {output['train_loss']:.4f}\n"
                  f"train acc: {output['train_acc']:.4f}\n"
                  f"train simple acc: {output['train_simple_acc']:.4f}\n"
                  f"valid loss: {output['valid_loss']:.4f}\n"
                  f"valid acc: {output['valid_acc']:.4f}\n"
                  f"valid simple acc: {output['valid_simple_acc']:.4f}"                  
                  )

        if hist_valid_loss[len(hist_valid_loss)-1] < best_valid_loss:
            best_valid_loss = hist_valid_loss[len(hist_valid_loss)-1]
            best_model_state = model.state_dict()
            
        
        plt.plot(hist_train_loss)
        plt.plot(hist_valid_loss)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.savefig(f'figure/loss_v3_10_w2_{fold+1}.png')
        plt.clf()
        plt.plot(hist_train_acc)
        plt.plot(hist_valid_acc)
        plt.xlabel('epoch')
        plt.ylabel('acc')
        plt.savefig(f'figure/acc_v3_10_w2_{fold+1}.png')
        plt.show()
        plt.clf()

    torch.save(best_model_state, 'model/best_model.pth')



    # 提出用ファイルの作成
    # 保存した重みを使うときは次のコードで
    model = VQAModel(vocab_size=len(train_dataset.question2idx)+1, n_answer=len(train_dataset.answer2idx)).to(device)
    model.load_state_dict(torch.load('model/best_model.pth'))
    model.eval()
    submission = []
    for image, question in test_loader:
        image, question = image.to(device), question.to(device)
        pred = model(image, question)
        pred = pred.argmax(1).cpu().item()
        submission.append(pred)

    print(submission)
    submission = [train_dataset.idx2answer[id] for id in submission]
    print(submission)
    submission = np.array(submission)
    # torch.save(model.state_dict(), "model.pth")
    np.save("submission/submission_v3_10.npy", submission)

if __name__ == "__main__":
    main()
