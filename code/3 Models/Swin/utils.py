import mediapipe as mp
import cv2
import numpy as np
import random
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import json

import torch
from torch.utils.data import Dataset

import torchvision
import torchvision.transforms.functional as F

    
class VideoDataset(Dataset):
    def __init__(self, video_labels, video_dir, IMG_SIZE, min_frame_count, classes, face_detection_model, ds_type = "test"):
        super(VideoDataset, self).__init__()
        self.video_labels = video_labels
        self.video_dir = video_dir
        self.IMG_SIZE = IMG_SIZE
        self.min_frame_count = min_frame_count
        self.video_data=[]
        self.video_label=[]
        self.score_threshold = 0.9
        self.nms_threshold = 0.3
        self.top_k = 5000
        self.scale = 1
        self.face_detection_model = face_detection_model
        

        for i in video_labels[video_labels.dataset==ds_type].index:
          frames, label = self.get_video_and_label(i)
          if frames is not None:  # В некоторых видео не может определить лицо mimetype?
              self.video_data.append(frames)
              self.video_label.append(classes.index(label))


    def __len__(self):
        return len(self.video_label)

    def crop_center_square(self, image, min_y, max_y, min_x, max_x):
        image = image[min_y : max_y, min_x : max_x]
        new_size = max(image.shape[0], image.shape[1])
        blank_image = np.zeros((new_size,new_size,3), np.uint8)
        blank_image[0 : image.shape[0], 0 : image.shape[1]] = image
        blank_image = cv2.resize(blank_image, (self.IMG_SIZE, self.IMG_SIZE))
        return blank_image


    def crop_coordinates(self, image):
      detector = cv2.FaceDetectorYN.create(
                  self.face_detection_model,
                  "",
                  (320, 320),
                  self.score_threshold,
                  self.nms_threshold,
                  self.top_k
              )
      shape = image.shape
      mp_face = mp.solutions.face_mesh
      with mp_face.FaceMesh(static_image_mode=True,
                                              max_num_faces=1,
                                              refine_landmarks=True,
                                              min_detection_confidence=0.5) as face_mesh:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        has_result = False
        i=0
        while not has_result:
          results = face_mesh.process(image)
          i+=1
          if i==5 or results.multi_face_landmarks is not None:
            has_result = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image.flags.writeable = True
        if results.multi_face_landmarks is None:
          img1Width = int(image.shape[1]*self.scale)
          img1Height = int(image.shape[0]*self.scale)

          tm = cv2.TickMeter()
          tm.start()
          detector.setInputSize((img1Width, img1Height))
          faces1 = detector.detect(image)
          tm.stop()
          if faces1[1] is not None:
            face_height = faces1[1][0][3]
            face_width = faces1[1][0][2]
            new_size = min(image.shape[0], image.shape[1], face_height*4)
            y_left =  faces1[1][0][1]
            x_left = faces1[1][0][0]
            min_x = max(int((x_left + face_width/2) - new_size/2),0)
            max_x = min(int((x_left + face_width/2) + new_size/2), image.shape[1])
            min_y = max(int((y_left + face_height) - new_size/2), 0)
            max_y = max(int((y_left + face_height) + new_size/2), image.shape[0])
            return min_y, max_y, min_x, max_x
          new_size = min(image.shape[:2])
          min_x = int(image.shape[1]/2 - new_size/2)
          max_x = int(image.shape[1]/2 + new_size/2)
          min_y = int(image.shape[0]/2 - new_size/2)
          max_y = int(image.shape[0]/2 + new_size/2)
          return min_y, max_y, min_x, max_x
        face_ls = results.multi_face_landmarks[0].landmark
        y_coordinates = [idx.y for idx in face_ls]
        x_coordinates = [idx.x for idx in face_ls]
        max_y, min_y = max(y_coordinates), min(y_coordinates)
        max_x, min_x = max(x_coordinates), min(x_coordinates)
        middle_x = (min_x + (max_x-min_x)/2) * shape[1]
        delta = max_y-min_y
        max_y, min_y = min(int((max_y+3*delta)*shape[0]),shape[0]), max(int((min_y-delta)*shape[0]),0)
        height = max_y - min_y
        max_x, min_x = min(int(max_x*shape[1]) + height//2, shape[1]) , max(int(min_x*shape[1]) - height//2,0)
      return min_y, max_y, min_x, max_x

    def load_video(self, path, begin, end, max_frames=0, label=None):
        cap = cv2.VideoCapture(path)
        frames = []
        #gray_frames = []
        min_y, max_y, min_x, max_x = 0, 0, 0, 0
        frame_index=begin+1
        while True and frame_index <= end:
          cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
          ret, frame = cap.read()
          if not ret:
            break
          if frame_index==begin+1:
            cc = self.crop_coordinates(frame)
            if cc is None:
             begin+=1
            else:
             min_y, max_y, min_x, max_x = cc
          if (min_y, max_y, min_x, max_x) != (0, 0, 0, 0):
            frame = self.crop_center_square(frame, min_y, max_y, min_x, max_x)
            frames.append(frame)
            # gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # gray_frames.append(gray_frame)
            frame_index+=1
          if len(frames) == max_frames:
           break

        cap.release()
        if len(frames)==0:
          print(f"No frames extracted from: {path}")
          return None
        totalt_frame_cnt = len(frames)
        if totalt_frame_cnt < self.min_frame_count:
          all_idxs = list(range(totalt_frame_cnt))
          add_idxs = random.choices(range(totalt_frame_cnt), k=self.min_frame_count-totalt_frame_cnt)
          all_idxs.extend(add_idxs)
        else:
          all_idxs = random.sample(range(totalt_frame_cnt), k=self.min_frame_count)
        #gray_frames = [gray_frames[i] for i in sorted(all_idxs)]
        frames = [frames[i] for i in sorted(all_idxs)]
        return torch.from_numpy(np.array(frames))

    def get_video_and_label(self, idx):
        filename  = os.path.join(self.video_dir, self.video_labels[self.video_labels.index==idx]['attachment_id'][idx]+".mp4")
        label = self.video_labels[self.video_labels.index==idx]['text'][idx]
        begin = self.video_labels[self.video_labels.index==idx]['begin'][idx]
        end = self.video_labels[self.video_labels.index==idx]['end'][idx]
        #print(f"Load {filename}")
        frames = self.load_video(filename, begin, end, label=label) # Загрузка видео!!!!
        return frames, label

    def __getitem__(self, index):
        frames= self.video_data[index]
        label=self.video_label[index]
        frames=torch.Tensor(frames)
        return frames.permute(3,0, 1,2), label
    
def validate_model(model, classes, epoch, criterion, optimizer, val_dataloader, device, best_acc, save=True, save_path=None, model_name=None):
    test_loss = list()
    correct=0
    total=0
    model.eval()
    for data, target in val_dataloader:
        data, target = data.to(device), target.to(device)
        with torch.no_grad():
            output = model(data/255)
        loss = criterion(output, target)
        test_loss.append(loss)
        pred = torch.argmax(output, 1)
        correct += (pred == target.item()).sum().float()
        total += len(target)
        predict_acc = correct / total
    if save and predict_acc >= best_acc:
        best_acc = predict_acc if predict_acc > best_acc else best_acc
        torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'classes': classes,
                    'epoch':epoch
                },
                    f'{save_path}/{model_name}-Val_acc-{predict_acc:.3f}.pth')
    return predict_acc, best_acc
  
def validate_model_batched(model, classes, epoch, criterion, optimizer, val_dataloader, device, best_acc, save=True, save_path=None, model_name=None):
    test_loss = list()
    correct=0
    total=0
    model.eval()
    for data, target in val_dataloader:
        data, target = data.to(device), target.to(device)
        with torch.no_grad():
            output = model(data/255)
        loss = criterion(output, target)
        test_loss.append(loss)
        pred = torch.argmax(output, 1)
        correct += (pred == target).sum().float()
        total += len(target)
        predict_acc = correct / total
    if save and predict_acc >= best_acc:
        best_acc = predict_acc if predict_acc > best_acc else best_acc
        torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'classes': classes,
                    'epoch':epoch
                },
                    f'{save_path}/{model_name}-Val_acc-{predict_acc:.3f}.pth')
    return predict_acc, best_acc
    
def train_model(model, optimizer, criterion, train_dataloader, device, scheduler=None, flip=False, perspective_transform=False):
    #import tqdm
    total_loss = []
    model.train()
    perspective_transformer = v2.RandomPerspective(distortion_scale=0.5, p=0.5)
    #pbar = tqdm(train_dataloader, desc=f'Train Epoch{epoch}/{epoches}')
    for data, target in train_dataloader:
        data, target = data.to(device), target.to(device)
        if flip:
            if random.sample([True, False],1)[0]:
                data = torch.flip(data, [4]) # Зеркальное отображение
        if perspective_transform:
            data = perspective_transformer(data)
        optimizer.zero_grad()  # Model Parameters Gradient Clear
        output = model(data/255)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss.append(loss.item())
        #pbar.set_description(f'Train Epoch:{epoch}/{epoches} train_loss:{round(np.mean(total_loss), 4)}')
    if scheduler is not None:
        scheduler.step()
    return round(np.mean(total_loss), 4)
            

def display_frames(n_frames, dataloader, classes):
    def show(imgs, label):
        if not isinstance(imgs, list):
            imgs = [imgs]
        fig, axs = plt.subplots(ncols=len(imgs), squeeze=False, figsize=(15, 15))
        plt.title(label)
        for i, img in enumerate(imgs):
            img = img.detach()
            img = F.to_pil_image(img)
            axs[0, i].imshow(np.asarray(img))
            axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    test_dl = iter(dataloader)
    for _ in range(n_frames):
        frames, labels = next(test_dl)
        frames = frames.permute(0,2,1,3,4)
        show(torchvision.utils.make_grid(frames[0]), classes[labels.item()])
        
def display_learning_dynamic(train_loss_dynamic, val_accuracy_dynamic, epochs, model_name):
    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.suptitle(f'{model_name}: Learning Dynamic')

    ax1.plot(epochs, train_loss_dynamic, 'o-')
    ax1.set_ylabel('Train loss')

    ax2.plot(epochs, val_accuracy_dynamic, '.-')
    ax2.set_xlabel('epoch')
    ax2.set_ylabel('Val accuracy')

    plt.show()
    
def classification_model_metrics(model, classes, dataloader, device, hm=True):
    TP={i:0 for i in range(len(classes))}
    FN={i:0 for i in range(len(classes))}
    FP={i:0 for i in range(len(classes))}
    total={i:0 for i in range(len(classes))}
    actual = list()
    predicted = list()
    model.eval()
    model.to
    for data, target in dataloader:
        data, target = data.to(device), target.to(device)
        with torch.no_grad():
            output = model(data/255)
        pred = torch.argmax(output, 1)
        prediction = pred.item()
        class_id = target.item()
        total[class_id] += 1
        TP[class_id] += class_id==prediction
        FN[class_id] += class_id!=prediction
        FP[pred.item()] += class_id!=prediction
        actual.append(class_id)
        predicted.append(prediction)
    accuracy = sum(TP.values())/sum(total.values())
    precision = np.mean([TP[i]/((TP[i]+FP[i]) or 1) for i in total.keys()])
    recall = np.mean([TP[i]/((TP[i]+FN[i]) or 1) for i in total.keys()])
    F1 = (2*precision*recall)/((precision+recall) or 1)
    cm = confusion_matrix(actual,predicted)
    if hm:
        sns.heatmap(cm,  annot=True, fmt='g', cmap=sns.color_palette("Blues", as_cmap=True))
        plt.ylabel('Prediction',fontsize=13)
        plt.xlabel('Actual',fontsize=13)
        plt.title('Confusion Matrix',fontsize=17)
        plt.show()
    print(f'Accuracy={accuracy}; Precision={precision}; Recall={recall}; F1={F1}')

def draw_dynamic(data, batch_sizes, n_classes, model_folder, model_name, type = 'val_accuracy_dynamic'):
  x = np.arange(1, 31, 1)
  title_dict = {'val_accuracy_dynamic': 'Accuracy dynamic',
                'train_loss_dynamic': 'Train Loss dynamic'}
  color_dict = {
      '3*10^(-2)': '#3E3E3E',
      '3*10^(-3)': '#D5E8F7',
      '3*10^(-4)': '#92C5EB',
      '3*10^(-5)': '#0072BC',
      '3*10^(-6)': '#FFAB40',
      '3*10^(-7)': '#5D5D5D',
      '3*10^(-8)': '#3E3E3E',
  } 

  linestyle_dict = {
      '3*10^(-2)': '--',
      '3*10^(-3)': '-',
      '3*10^(-4)': '-',
      '3*10^(-5)': '-',
      '3*10^(-6)': '-',
      '3*10^(-7)': '-',
      '3*10^(-8)': '-',
  }

  def upd_len(some_list, target_size=30):
    cl = len(some_list)
    to_add = [0]*(target_size-cl)
    return some_list + to_add

  fig, (ax0, ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=4, sharex=True, figsize=(24, 4), sharey=True)
    
  if type == 'val_accuracy_dynamic':
    ax0.set_ylim(0, 1)
  ax0.set_title(f"batch_size = {batch_sizes[0]}")
  ax1.set_title(f"batch_size = {batch_sizes[1]}")
  ax2.set_title(f"batch_size = {batch_sizes[2]}")
  ax3.set_title(f"batch_size = {batch_sizes[3]}")


  for example in data:
    with open(model_folder + example['filename'], 'r') as f:
      dynamic = json.load(f)
      if example['bs'] == batch_sizes[0]:
        ax0.plot(x, upd_len(dynamic[type]), label = f"lr={example['lr']}", color = color_dict[example['lr']], linestyle = linestyle_dict[example['lr']])
      elif example['bs'] == batch_sizes[1]:
        ax1.plot(x, upd_len(dynamic[type]), label = f"lr={example['lr']}", color = color_dict[example['lr']], linestyle = linestyle_dict[example['lr']])
      elif example['bs'] == batch_sizes[2]:
        ax2.plot(x, upd_len(dynamic[type]), label = f"lr={example['lr']}", color = color_dict[example['lr']], linestyle = linestyle_dict[example['lr']])
      elif example['bs'] == batch_sizes[3]:
        ax3.plot(x, upd_len(dynamic[type]), label = f"lr={example['lr']}", color = color_dict[example['lr']], linestyle = linestyle_dict[example['lr']])

  fig.suptitle(f"{model_name}: {title_dict[type]} ({n_classes} classes)")
  ax0.legend()
  ax1.legend()
  ax2.legend()
  ax3.legend()
  plt.show()
