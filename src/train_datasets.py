import torch
from torch.utils.data import Dataset
from datasets import load_dataset, concatenate_datasets, DatasetDict
from typing import List, Tuple, Dict
from transformers import T5Tokenizer, MT5ForConditionalGeneration, AdamW

class TedTalksDataset(Dataset):
    def __init__(self, dataset_name: str,
                 tokenizer: T5Tokenizer,
                 max_length: int = 128,
                 language_pair: Tuple[str, str] = ("en", "lt"),
                 years: List[int] = [2014, 2015, 2016]):
        if dataset_name != "ted_talks_iwslt":
            raise ValueError("Dataset must be ted_talks_iwslt")

        self.tokenizer = tokenizer
        self.dataset = self.load_and_combine_datasets(dataset_name, language_pair, years)
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
      item = self.dataset[idx]
      source_text = item["en"]
      target_text = item["lt"]

      source_promt = f"translate English to Lithuanian: {source_text}"

      source_encoding = self.tokenizer(source_promt,
                                       padding="max_length",
                                       truncation=True,
                                       max_length=self.max_length,
                                       return_tensors="pt")

      target_encoding = self.tokenizer(target_text,
                                       padding="max_length",
                                       truncation=True,
                                       max_length=self.max_length,
                                       return_tensors="pt")

      return {
          "input_ids": source_encoding["input_ids"].squeeze(),
          "attention_mask": source_encoding["attention_mask"].squeeze(),
          "labels": target_encoding["input_ids"].squeeze()
      }

    def load_and_combine_datasets(self, dataset_name: str, language_pair: Tuple[str, str], years: List[int]) -> DatasetDict:
      combined_datasets = None

      for year in years:
          # Загрузка датасета за каждый год
          data = load_dataset(dataset_name, language_pair=language_pair, year=str(year), trust_remote_code=True)

          data = data.map(lambda example: {'translation': {language_pair[0]: example['translation'][language_pair[0]],
                                                          language_pair[1]: example['translation'][language_pair[1]]}},
                          remove_columns=['translation'])

          # Объединение датасетов
          if combined_datasets is None:
              combined_datasets = data
          else:
              combined_datasets = {split: concatenate_datasets([combined_datasets[split], data[split]]) for split in combined_datasets.keys()}

      return combined_datasets['train']['translation']

class PontoonTranslationsDataset(Dataset):
    def __init__(self, dataset_name: str,
                 tokenizer: T5Tokenizer,
                 max_length: int = 128,
                 language_pair: Tuple[str, str] = ("en", "lt")):
        if dataset_name != "ayymen/Pontoon-Translations":
            raise ValueError("Dataset must be ayymen/Pontoon-Translations")

        self.tokenizer = tokenizer
        self.dataset = load_dataset("ayymen/Pontoon-Translations", f"{language_pair[0]}-{language_pair[1]}")
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset["train"])

    def __getitem__(self, idx):
      item = self.dataset["train"][idx]
      source_text = item["source_string"]
      target_text = item["target_string"]

      source_promt = f"translate English to Lithuanian: {source_text}"

      source_encoding = self.tokenizer(source_promt,
                                       padding="max_length",
                                       truncation=True,
                                       max_length=self.max_length,
                                       return_tensors="pt")

      target_encoding = self.tokenizer(target_text,
                                       padding="max_length",
                                       truncation=True,
                                       max_length=self.max_length,
                                       return_tensors="pt")

      return {
          "input_ids": source_encoding["input_ids"].squeeze(),
          "attention_mask": source_encoding["attention_mask"].squeeze(),
          "labels": target_encoding["input_ids"].squeeze()
      }

#@title Dataset class
class ScorisMergedDataset(Dataset):
    def __init__(self, dataset_name: str,
                 tokenizer: T5Tokenizer,
                 max_length: int = 128,):
        if dataset_name != "scoris/en-lt-merged-data":
            raise ValueError("Dataset must be scoris/en-lt-merged-data")

        self.tokenizer = tokenizer
        self.dataset = load_dataset(dataset_name,).remove_columns(['__index_level_0__'])
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset["train"])

    def __getitem__(self, idx):
      item = self.dataset["train"][idx]["translation"]
      source_text = item['en']
      target_text = item["lt"]

      source_promt = f"translate English to Lithuanian: {source_text}"

      source_encoding = self.tokenizer(source_promt,
                                       padding="max_length",
                                       truncation=True,
                                       max_length=self.max_length,
                                       return_tensors="pt")

      target_encoding = self.tokenizer(target_text,
                                       padding="max_length",
                                       truncation=True,
                                       max_length=self.max_length,
                                       return_tensors="pt")

      return {
          "input_ids": source_encoding["input_ids"].squeeze(),
          "attention_mask": source_encoding["attention_mask"].squeeze(),
          "labels": target_encoding["input_ids"].squeeze()
      }

class MergedDataset(Dataset):
    def __init__(self, dataset_name: str,
                 tokenizer: T5Tokenizer,
                 max_length: int = 128):
        if dataset_name != "ALL":
            raise ValueError("this class is under development")

        self.tokenizer = tokenizer

        self.lithuanian_translations = load_dataset("werent4/lithuanian-translations", "en-lt")

        self.ted_talks_iwslt = self.load_and_combine_datasets(dataset_name ="ted_talks_iwslt", language_pair = ("en", "lt"), years= [2014, 2015, 2016])
        self.ted_talks_iwslt = self.ted_talks_iwslt.map(lambda x: {'en': x['translation']['en'], 'lt': x['translation']['lt']}, remove_columns=['translation'])

        self.scoris = load_dataset("scoris/en-lt-merged-data",).remove_columns(['__index_level_0__'])
        self.scoris = self.scoris['train'].map(lambda x: {'en': x['translation']['en'], 'lt': x['translation']['lt']}, remove_columns=["translation"])

        self.dataset = concatenate_datasets([self.lithuanian_translations['train'], self.ted_talks_iwslt,
                                             self.scoris])
        self.max_length = max_length

    def __len__(self):
        return 2 * len(self.dataset)

    def __getitem__(self, idx):
        original_idx = idx // 2
        #print(original_idx)
        reverse = idx % 2 == 1
        #print(reverse)

        item = self.dataset[original_idx]

        if not reverse:
            source_text = item['en']
            target_text = item["lt"]
            source_prompt = f"<EN2LT> {source_text}"
            #print(f"{source_prompt} - {target_text}")
        else:
            source_text = item['lt']
            target_text = item["en"]
            source_prompt = f"<LT2EN> {source_text}"
            #print(f"{source_prompt} - {target_text}")

        source_encoding = self.tokenizer(source_prompt,
                                         padding="max_length",
                                         truncation=True,
                                         max_length=self.max_length,
                                         return_tensors="pt")

        target_encoding = self.tokenizer(target_text,
                                         padding="max_length",
                                         truncation=True,
                                         max_length=self.max_length,
                                         return_tensors="pt")

        return {
            "input_ids": source_encoding["input_ids"].squeeze(),
            "attention_mask": source_encoding["attention_mask"].squeeze(),
            "labels": target_encoding["input_ids"].squeeze()
        }

    def load_and_combine_datasets(self, dataset_name: str, language_pair: Tuple[str, str], years: List[int]) -> DatasetDict:
      combined_datasets = None

      for year in years:
          data = load_dataset(dataset_name, language_pair=language_pair, year=str(year), trust_remote_code=True)

          data = data.map(lambda example: {'translation': {language_pair[0]: example['translation'][language_pair[0]],
                                                          language_pair[1]: example['translation'][language_pair[1]]}},
                          remove_columns=['translation'])

          if combined_datasets is None:
              combined_datasets = data
          else:
              combined_datasets = {split: concatenate_datasets([combined_datasets[split], data[split]]) for split in combined_datasets.keys()}

      return combined_datasets['train']

