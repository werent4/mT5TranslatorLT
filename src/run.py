import argparse, os
import torch
from torch.utils.data import DataLoader
from transformers import T5Tokenizer, MT5ForConditionalGeneration, AdamW

from train_datasets import TedTalksDataset, PontoonTranslationsDataset, ScorisMergedDataset, MergedDataset
from train import train_model

def main(args):
    torch.manual_seed(42)

    tokenizer = T5Tokenizer.from_pretrained(args.tokenizer)
    tokenizer.add_tokens(["<EN2LT>", "<LT2EN>"])
    model = MT5ForConditionalGeneration.from_pretrained(args.model)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    if args.dataset =="scoris/en-lt-merged-data":
        dataset = ScorisMergedDataset("scoris/en-lt-merged-data",
                            tokenizer = tokenizer,
                            max_length = 128,
                            )
    elif args.dataset =="ted_talks_iwslt":
        dataset = TedTalksDataset("ted_talks_iwslt",
                            tokenizer = tokenizer,
                            max_length = 128,
                            )
    elif args.dataset =="ayymen/Pontoon-Translations":
        dataset = PontoonTranslationsDataset("ayymen/Pontoon-Translations",
                            tokenizer = tokenizer,
                            max_length = 128,
                            )
    elif args.dataset =="ALL":
        dataset = MergedDataset("ALL",
                            tokenizer = tokenizer,
                            max_length = 128,
                            )
    else:
        raise ValueError(f"Dataset: {args.dataset} not supported")

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay= args.weight_decay)

    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)
    tokenizer_dir = output_dir+ "/tokeniser"
    os.makedirs(tokenizer_dir, exist_ok=True)

    save_model_path = f"{output_dir}/mT5Translator_scoris_en_lt_updated.pth"
    training_info_path = f"{output_dir}/training_info_scoris_en_lt.json"

    tokenizer.save_pretrained(tokenizer_dir)
    print("\nTokenizer saved.")

    train_model(
        model= model,
        dataloader= loader,
        optimizer = optimizer,
        num_epochs= args.num_epochs,
        save_path= save_model_path,
        start_from_batch= args.start_from_batch,
        training_info_path= training_info_path,
        save_each_steps= args.save_each_steps,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train mT5 model for translation task")
    parser.add_argument('--tokenizer', type=str, default='google/mt5-small', required= True, help='Path to the tokenizer')
    parser.add_argument('--model', type=str, default='werent4/mt5TranslatorLT', required= True, help='Path to the model')
    parser.add_argument('--dataset', type=str, default='scoris/en-lt-merged-data', required= True, help='Path to the dataset')
    parser.add_argument('--max_length', type=int, default=128, required= True, help='Maximum sequence length')
    parser.add_argument('--batch_size', type=int, default=14, required= True, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-5, required= True, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, required= True, help='Weight decay')
    parser.add_argument('--num_epochs', type=int, default=1, required= True, help='Number of epochs')
    parser.add_argument('--start_from_batch', type=int, default=0, required= True, help='Start training from the specified batch')
    parser.add_argument('--save_each_steps', type=int, default=15, required= True, help='Save the model every N steps')

    args = parser.parse_args()
    main(args)