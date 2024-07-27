
from data.dataset import readIndex, dataReadPip, loadedDataset
from model.deepcrack import DeepCrack
from t import DeepCrackTrainer
import cv2
from tqdm import tqdm
import numpy as np
import torch
import os


def test(test_data_path='dataset/test_img',
         val_data_path='dataset/test_lab',
         save_path='results/',
         pretrained_model='checkpoints/hlcf_crack.pth', ):
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    test_pipline = dataReadPip(transforms=None)
    test_list = readIndex(test_data_path,val_data_path)
    print(test_list)

    test_dataset = loadedDataset(test_list, preprocess=test_pipline)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1,
                                              shuffle=False, num_workers=1, drop_last=False)

    # -------------------- build trainer --------------------- #
    device = torch.device("cuda")
    num_gpu = torch.cuda.device_count()

    model = DeepCrack()
    model = torch.nn.DataParallel(model, device_ids=range(num_gpu))
    model.to(device)

    trainer = DeepCrackTrainer(model).to(device)
    model.load_state_dict(trainer.saver.load(pretrained_model, multi_gpu=True))
    model.eval()

    torch.cuda.empty_cache()  # Clear cache before starting

    with torch.no_grad():
        for names, (img, lab) in tqdm(zip(test_list, test_loader)):
            test_data = img.type(torch.cuda.FloatTensor).to(device)
            test_target = lab.type(torch.cuda.FloatTensor).to(device)

            test_pred = trainer.val_op(test_data, test_target)

            test_pred = torch.sigmoid(test_pred[0].cpu().squeeze())
            print("test_pred", test_pred.shape)

            save_pred = torch.zeros((512, 512))
            save_pred[:512, :] = test_pred
            # save_pred[512:, :] = lab.cpu().squeeze()

            save_name = os.path.join(save_path, os.path.split(names[1])[1])
            save_pred = save_pred.numpy() * 255
            cv2.imwrite(save_name, save_pred.astype(np.uint8))

            # Clear memory after each iteration
            del test_data, test_target, test_pred, save_pred
            torch.cuda.empty_cache()


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    test()
