from pathlib import Path
import json
from torch.cuda.amp import autocast
import torch
from tqdm import tqdm
from typing import List, Tuple

def create_checkpoint(save_path: Path, # Model save patj
                      training_info_path: Path, # Training info path
                      processed_len: int, # processed batches
                      total_batches_processed: int, # total batches processed
                      model, # model
                      total_loss, # losses on batch
                      ):
  # Save the model state
  checkpoint_path = save_path
  torch.save(model.state_dict(), checkpoint_path)
  print("\nModel saved, batch: ", total_batches_processed)
  # Save batch info to JSON
  batch_info = {
    "batch_number": total_batches_processed,
    "total_batches_processed": processed_len,
    "loss": total_loss / processed_len
  }
  with open(training_info_path, 'w') as json_file:
    json.dump(batch_info, json_file)
  print(f"Batch information saved to {training_info_path}")

def train_model(model,
                dataloader,
                optimizer,
                num_epochs,
                save_path: Path,
                training_info_path: Path,
                start_from_batch: int=350,
                save_each_steps: int = 2000,
                ):
  total_batches_skipped = 0
  total_batches_processed = 0

  model.train()
  for epoch in range(num_epochs):
      total_loss = 0
      torch.cuda.empty_cache()

      for batch in tqdm(dataloader, desc= "Processing dataset"):
          #Skipping batches to nessesary point
          if total_batches_processed < start_from_batch:
            total_batches_processed += 1
            continue

          input_ids = batch['input_ids'].to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
          attention_mask = batch['attention_mask'].to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
          labels = batch['labels'].to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

          outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
          loss = outputs.loss

          loss.backward()
          optimizer.step()
          optimizer.zero_grad()

          total_loss += loss.item()
          print(f"Batch Loss: {loss.item()}")

          total_batches_processed += 1
          if total_batches_processed % save_each_steps == 0:
            processed_len = total_batches_processed - start_from_batch
            create_checkpoint(
                save_path=save_path,
                training_info_path=training_info_path,
                processed_len=processed_len,
                total_batches_processed=total_batches_processed,
                model=model,
                total_loss=total_loss
            )

      print(f"Epoch {epoch+1}, Average Loss: {total_loss / len(dataloader)}")

  torch.save(model.state_dict(), save_path)

