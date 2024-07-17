import re
import random
import time
from statistics import mode

from PIL import Image
import numpy as np
import pandas
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torch import einsum
from einops import rearrange
from einops_exts import check_shape, rearrange_many
import torchvision
from torchvision import transforms
from transformers import BertModel, BertTokenizer
from tqdm import tqdm
from datasets import load_dataset
import os
from torchvision.transforms import ToTensor

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

# Hugging FaceのVizWizクラスマッピングを使用
class_mapping_df = pandas.read_csv('/root/VQA/mapping.txt', header=None, names=['answer', 'class_id'])

answer2idx = {row['answer']: row['class_id'] for _, row in class_mapping_df.iterrows()}
idx2answer = {row['class_id']: row['answer'] for _, row in class_mapping_df.iterrows()}


# 1. データローダーの作成
class VQADataset(torch.utils.data.Dataset):
    def __init__(self, df_path, image_dir, transform=None, answer=True):
        self.transform = transform  # 画像の前処理
        self.image_dir = image_dir  # 画像ファイルのディレクトリ
        self.df = pandas.read_json(df_path)  # 質問ファイルのパス，question, answerを持つDataFrame
        self.answer = answer

        # question / answerの辞書を作成
        self.question2idx = {}
        self.answer2idx = {}
        self.idx2question = {}
        self.idx2answer = {}

        self.answer2idx = answer2idx
        self.idx2answer = idx2answer

        # 質問文に含まれる単語を辞書に追加
        for question in self.df["question"]:
            question = process_text(question)
            words = question.split(" ")
            for word in words:
                if word not in self.question2idx:
                    self.question2idx[word] = len(self.question2idx)
        self.idx2question = {v: k for k, v in self.question2idx.items()}  # 逆変換用の辞書(question)

        if self.answer:
            # 回答に含まれる単語を辞書に追加
            for answers in self.df["answers"]:
                for answer in answers:
                    word = answer["answer"]
                    word = process_text(word)
                    if word not in self.answer2idx:
                        self.answer2idx[word] = len(self.answer2idx)
            self.idx2answer = {v: k for k, v in self.answer2idx.items()}  # 逆変換用の辞書(answer)

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
            質問文をone-hot表現に変換したもの
        answers : torch.Tensor  (n_answer)
            10人の回答者の回答のid
        mode_answer_idx : torch.Tensor  (1)
            10人の回答者の回答の中で最頻値の回答のid
        """
        image = Image.open(f"{self.image_dir}/{self.df['image'][idx]}")
        image = self.transform(image)
        image = image/255.0

        question_text = self.df["question"][idx]
        question_text = process_text(question_text)
        
        """
        question = np.zeros(len(self.idx2question) + 1)  # 未知語用の要素を追加
        question_words = self.df["question"][idx].split(" ")
        for word in question_words:
            try:
                question[self.question2idx[word]] = 1  # one-hot表現に変換
            except KeyError:
                question[-1] = 1  # 未知語
        """
        if self.answer:
            answers = [self.answer2idx[process_text(answer["answer"])] for answer in self.df["answers"][idx]]
            mode_answer_idx = mode(answers)  # 最頻値を取得（正解ラベル）

            return image, question_text, torch.Tensor(answers), int(mode_answer_idx)

        else:
            return image, question_text

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


# 3. モデルのの実装
# SpatialAttention
class SpatialAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 8):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv)

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        out = self.to_out(out)
        return out
    
# ResNet with Attention
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        self.attention = SpatialAttention(in_channels)
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
        atten_x = self.attention(x)
        atten_x += residual

        residual = atten_x
        out = self.relu(self.bn1(self.conv1(atten_x)))
        out = self.bn2(self.conv2(out))

        out += self.shortcut(residual)
        out = self.relu(out)

        return out


class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        self.attention = SpatialAttention(in_channels)
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
        atten_x = self.attention(x)
        atten_x += residual

        residual = atten_x
        out = self.relu(self.bn1(self.conv1(atten_x)))
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



#BERT embeding
class BertEmbeding(nn.Module):
    def __init__(self, pretrained_model_name='huawei-noah/TinyBERT_General_4L_312D', device='cpu'):
        super().__init__()
        self.device = device
        self.bert = BertModel.from_pretrained(pretrained_model_name).to(device)
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)

    def forward(self, questions):
        inputs = self.tokenizer(questions, return_tensors='pt', padding=True, truncation=True).to(self.device)
        outputs = self.bert(**inputs)
        pooled_output = outputs.last_hidden_state.mean(1)
        return pooled_output


class SelfAttention(nn.Module):
    def __init__(self, hidden_dim, heads):
        super().__init__()
        self.heads = heads
        self.head_dim = hidden_dim // heads
        assert hidden_dim % heads == 0, "hidden_dim must be divisible by num_heads"

        self.scale = self.head_dim ** -0.5
        self.to_qkv = nn.Linear(hidden_dim, hidden_dim * 3, bias=False)
        self.to_out = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        q = q * self.scale
        attn = torch.einsum('bhid,bhjd->bhij', q, k)
        attn = attn.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out
    
class MultiModalAttention(nn.Module):
    def __init__(self, img_dim, q_dim, hidden_dim, heads=16):
        super().__init__()
        self.heads = heads
        self.hidden_dim = hidden_dim

        assert hidden_dim % heads == 0, "hidden_dim must be divisible by num_heads"

        self.feat1_fc = nn.Linear(img_dim, hidden_dim)
        self.feat2_fc = nn.Linear(q_dim, hidden_dim)
        self.attention = SelfAttention(hidden_dim=hidden_dim, heads=heads)
        self.norm = nn.LayerNorm(hidden_dim * 2)
        self.dropout = nn.Dropout(0.1)

    def forward(self, features1, features2):
        feat1_proj = self.feat1_fc(features1).unsqueeze(1)  # [batch, 1, hidden_dim]
        feat2_proj = self.feat2_fc(features2).unsqueeze(1)  # [batch, 1, hidden_dim]
        combined = torch.cat((feat1_proj, feat2_proj), dim=1)  # [batch, 2, hidden_dim]
        attended_features = self.attention(combined)

        attended_features = attended_features.view(attended_features.size(0), -1)  # [batch, hidden_dim * 2]
        attended_features = self.norm(attended_features)
        attended_features = self.dropout(attended_features)

        res = torch.cat([feat1_proj.squeeze(1), feat2_proj.squeeze(1)], dim=1)

        return attended_features + res #[batch, hidden_dim*2]
    

class VQAModel(nn.Module):
    def __init__(self, n_answer: int):
        super().__init__()
        self.resnet = ResNet50() #out_dim = 512
        
        self.multi_attention = MultiModalAttention(img_dim=512, q_dim=312, hidden_dim=64)
        self.self_attention = SelfAttention(hidden_dim=64*2, heads=16)
        self.fc = nn.Sequential(
            nn.Linear(64*2, 64*4),
            nn.ReLU(inplace=True),
            nn.Linear(64*4, 64*8),
            nn.ReLU(inplace=True),
            nn.Linear(64*8, n_answer)
        )

    def forward(self, image, question):
        image_feature = self.resnet(image)  # 画像の特徴量
        res = self.multi_attention(image_feature, question)
        x = self.self_attention(res.unsqueeze(1)).squeeze(1)
        x = x + res
        x = self.fc(x)

        return x


# 4. 学習の実装
def train(model, dataloader, optimizer, criterion, device, bert):
    model.train()

    total_loss = 0
    total_acc = 0
    simple_acc = 0

    start = time.time()
    for image, question, answers, mode_answer in tqdm(dataloader, desc="Training"):
        question = bert(question)
        image, question, answer, mode_answer = image.to(device), question.to(device), answers.to(device), mode_answer.to(device)

        pred = model(image, question)
        loss = criterion(pred, mode_answer.squeeze())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()

        total_loss += loss.item()
        total_acc += VQA_criterion(pred.argmax(1), answers)  # VQA accuracy
        simple_acc += (pred.argmax(1) == mode_answer).float().mean().item()  # simple accuracy

    return total_loss / len(dataloader), total_acc / len(dataloader), simple_acc / len(dataloader), time.time() - start


def eval(model, dataloader, optimizer, criterion, device, bert):
    model.eval()

    total_loss = 0
    total_acc = 0
    simple_acc = 0

    start = time.time()
    test_size = 100
    c = 0
    for image, question, answers, mode_answer in dataloader:
        question = bert(question)
        image, question, answer, mode_answer = image.to(device), question.to(device), answers.to(device), mode_answer.to(device)

        pred = model(image, question)
        loss = criterion(pred, mode_answer.squeeze())
        torch.cuda.empty_cache()

        total_loss += loss.item()
        total_acc += VQA_criterion(pred.argmax(1), answers)  # VQA accuracy
        simple_acc += (pred.argmax(1) == mode_answer).mean().item()  # simple accuracy

        c += 1
        if c == test_size:
            break

    return total_loss / test_size, total_acc / test_size, simple_acc / test_size, time.time() - start


def main():
    # deviceの設定
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    # dataloader / model
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    train_dataset = VQADataset(df_path="/root/VQA/train.json", image_dir="/root/VQA/train", transform=transform)
    test_dataset = VQADataset(df_path="/root/VQA/valid.json", image_dir="/root/VQA/valid/valid", transform=transform, answer=False)
    test_dataset.update_dict(train_dataset)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = VQAModel(n_answer=len(train_dataset.answer2idx)).to(device)

    # optimizer / criterion
    num_epoch = 20
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)

    bert = BertEmbeding(device=device)

    # train model
    print('check')
    for epoch in range(num_epoch):
        train_loss, train_acc, train_simple_acc, train_time = train(model, train_loader, optimizer, criterion, device, bert)
        print(f"【{epoch + 1}/{num_epoch}】\n"
              f"train time: {train_time:.2f} [s]\n"
              f"train loss: {train_loss:.4f}\n"
              f"train acc: {train_acc:.4f}\n"
              f"train simple acc: {train_simple_acc:.4f}")
        """
        test_loss, test_acc, test_simple_acc, test_time = eval(model, test_loader, optimizer, criterion, device, bert)
        print(f"【{epoch + 1}/{num_epoch}】\n"
              f"test time: {test_time:.2f} [s]\n"
              f"test loss: {test_loss:.4f}\n"
              f"test acc: {test_acc:.4f}\n"
              f"test simple acc: {test_simple_acc:.4f}\n")
        """

    # 提出用ファイルの作成
    model.eval()
    submission = []
    for image, question in tqdm(test_loader, desc='testing'):
        question = bert(question)
        image, question = image.to(device), question.to(device)
        pred = model(image, question)
        pred = pred.argmax(1).cpu().item()
        submission.append(pred)

    submission = [train_dataset.idx2answer[id] for id in submission]
    submission = np.array(submission)
    torch.save(model.state_dict(), f"/root/VQA/model.pth")
    np.save(f"/root/VQA/submission.npy", submission)

if __name__ == "__main__":
    main()
