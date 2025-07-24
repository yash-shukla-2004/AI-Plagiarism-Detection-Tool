import torch
from old_app2 import model

# Save the model's state_dict
def save_model(model, file_path):
    torch.save(model.state_dict(), file_path)
    print(f"Model saved to {file_path}")



