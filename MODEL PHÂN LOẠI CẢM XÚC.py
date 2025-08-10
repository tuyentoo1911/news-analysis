import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datasets import Dataset
from underthesea import word_tokenize
import logging

# 1. Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('training.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# 2. Hàm load và tiền xử lý dữ liệu
def load_and_preprocess_data(file_path):
    try:
        logger.info(f"Đang đọc dữ liệu từ {file_path}")
        df = pd.read_csv(file_path)
        
        # Kiểm tra cấu trúc dữ liệu
        assert 'text' in df.columns, "File CSV phải có cột 'text'"
        assert 'label' in df.columns, "File CSV phải có cột 'label'"
        
        # Tạo ánh xạ nhãn
        label2id = {label: idx for idx, label in enumerate(df['label'].unique())}
        id2label = {v: k for k, v in label2id.items()}
        df['label'] = df['label'].map(label2id)
        
        # Tiền xử lý văn bản
        logger.info("Đang tiền xử lý văn bản...")
        df['text'] = df['text'].apply(lambda x: word_tokenize(str(x), format="text"))
        
        return df, label2id, id2label
        
    except Exception as e:
        logger.error(f"Lỗi khi xử lý dữ liệu: {str(e)}")
        raise

# 3. Hàm tokenize dữ liệu
def tokenize_data(dataset, tokenizer):
    try:
        logger.info("Đang tokenize dữ liệu...")
        def tokenize_fn(batch):
            return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)
        
        dataset = dataset.map(tokenize_fn, batched=True)
        dataset = dataset.rename_column("label", "labels")
        dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
        
        return dataset
    except Exception as e:
        logger.error(f"Lỗi khi tokenize: {str(e)}")
        raise

# 4. Hàm tính metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc}

# 5. Hàm huấn luyện mô hình
def train_model(train_dataset, test_dataset, label2id, id2label):
    try:
        logger.info("Đang khởi tạo mô hình...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)
        model = AutoModelForSequenceClassification.from_pretrained(
            "vinai/phobert-base",
            num_labels=len(label2id),
            id2label=id2label,
            label2id=label2id
        ).to(device)
        
        training_args = TrainingArguments(
            output_dir="./results",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=3,
            weight_decay=0.01,
            logging_dir="./logs",
            logging_steps=50,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=compute_metrics
        )
        
        logger.info("Bắt đầu huấn luyện...")
        trainer.train()
        trainer.save_model("./phobert_sentiment")
        logger.info("Huấn luyện hoàn tất!")
        
        return trainer, tokenizer
        
    except Exception as e:
        logger.error(f"Lỗi khi huấn luyện: {str(e)}")
        raise

# 6. Hàm dự đoán
class SentimentAnalyzer:
    def __init__(self, model_path, tokenizer):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device)
        self.tokenizer = tokenizer
        
    def predict(self, text):
        try:
            processed_text = word_tokenize(str(text), format="text")
            inputs = self.tokenizer(
                processed_text, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=128
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                predicted_class = torch.argmax(outputs.logits, dim=1).item()
            
            return self.model.config.id2label[predicted_class]
        
        except Exception as e:
            logger.error(f"Lỗi khi dự đoán: {str(e)}")
            return None

# 7. Hàm main
def main():
    try:
        # Load và tiền xử lý dữ liệu
        df, label2id, id2label = load_and_preprocess_data("sentiment_dataset.csv")
        
        # Chia dữ liệu
        train_df, test_df = train_test_split(
            df, 
            test_size=0.2, 
            random_state=42, 
            stratify=df['label']
        )
        
        # Tạo Dataset
        train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
        test_dataset = Dataset.from_pandas(test_df.reset_index(drop=True))
        
        # Huấn luyện mô hình
        trainer, tokenizer = train_model(train_dataset, test_dataset, label2id, id2label)
        
        # Khởi tạo bộ phân tích
        analyzer = SentimentAnalyzer("./phobert_sentiment", tokenizer)
        
        # Test mẫu
        test_samples = [
            "Sản phẩm này rất tốt, tôi rất hài lòng",
            "Dịch vụ tệ hại, sẽ không quay lại",
            "Bình thường, không có gì đặc biệt"
        ]
        
        logger.info("\nKết quả dự đoán mẫu:")
        for sample in test_samples:
            result = analyzer.predict(sample)
            print(f"Text: {sample}")
            print(f"→ Cảm xúc: {result}\n")
            
    except Exception as e:
        logger.error(f"Lỗi trong quá trình chạy: {str(e)}")

if __name__ == "__main__":
    main()