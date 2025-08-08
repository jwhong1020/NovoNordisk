import os
import torch
from torchvision.utils import save_image
from tqdm import tqdm

from anomaly_models import Generator
from anomaly_datasets import get_loader  # 데이터셋 로더 함수 (test_dataset.py 참고)

# ===== 설정 =====
normal_images_dir = "OCT2017/OCT2017/test/NORMAL"  # Normal OCT test 이미지 경로
output_dir = "results_normal2normal"               # 출력 저장 경로
checkpoint_path = "checkpoints/G_BA.pth"           # 학습된 G_BA 경로
image_size = 256                                   # 모델 입력 크기
batch_size = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(output_dir, exist_ok=True)

# ===== 모델 로드 =====
print(f"Loading G_BA from {checkpoint_path}...")
G_BA = Generator(input_nc=1, output_nc=1).to(device)
G_BA.load_state_dict(torch.load(checkpoint_path, map_location=device))
G_BA.eval()

# ===== 데이터 로드 =====
loader = get_loader(
    image_path=normal_images_dir,
    image_size=image_size,
    batch_size=batch_size,
    mode='test',
    num_workers=4
)

# ===== Normal → Normal 변환 =====
with torch.no_grad():
    for i, (imgs, _) in enumerate(tqdm(loader, desc="Normal→Normal")):
        imgs = imgs.to(device)
        outputs = G_BA(imgs)  # 이미 Normal → Normal
        save_image(outputs, os.path.join(output_dir, f"{i:04d}_reconstructed.png"))
        save_image(imgs, os.path.join(output_dir, f"{i:04d}_input.png"))

print(f"Done! Results saved in {output_dir}")
