


'''Semisupervised model all layer embedding regularization'''


import torch
import torch.nn as nn
import scipy
from grae.data.base_dataset import DEVICE
device = DEVICE

class SemiSupModuleAllR(nn.Module):
    def __init__(self, 
                 input_dim, 
                 n_classes,
                 z_dim,
                 hidden1 = 100,
                 hidden3 = 100,
                 hidden2 = 100):
        
        super().__init__()

        self.prediction = False
        self.fc1 =  nn.Linear(input_dim, hidden1)  
        self.batchnorm1 = nn.BatchNorm1d(hidden1)
        self.fc2 =  nn.Linear(hidden1, hidden2) 
        self.batchnorm2 = nn.BatchNorm1d(hidden2)
        self.fc3 =  nn.Linear(hidden2, hidden3) 
        self.batchnorm3 = nn.BatchNorm1d(hidden3)
        self.fc4 =  nn.Linear(hidden3, n_classes)  
        self.relu = nn.ReLU()
        self.reg1 = nn.Linear(hidden1, z_dim) 
        self.reg2 = nn.Linear(hidden2, z_dim) 
        self.reg3 = nn.Linear(hidden3, z_dim) 
        self.reg4 = nn.Linear(n_classes, z_dim) 
        self.softmax = nn.Softmax()
        
    def forward(self, x):
        x1 = self.batchnorm1(self.relu(self.fc1(x)))
        R1 = self.reg1(x1)
        
        x2 = self.batchnorm2(self.relu(self.fc2(x1)))
        R2 = self.reg2(x2)
        
        x3 = self.batchnorm3(self.relu(self.fc3(x2)))
        R3 = self.reg3(x3)
        
        y_pred = self.softmax(self.fc4(x3))
        R4 = self.reg4(y_pred)
        
        return y_pred, R1, R2, R3, R4

# Hyperparameters defaults
BATCH_SIZE = 128
LR = .0001
WEIGHT_DECAY = 0
EPOCHS = 200
       
        
class semisupervised_all3():
        def __init__(self, *, lr=LR, 
                     epochs=EPOCHS, 
                     batch_size=BATCH_SIZE, 
                     weight_decay = 0,
                     Embedding = False,
                     lam = 1,
                     regression = False):
                

            self.fitted = False
            self.torch_module = None
            self.optimizer = None
            self.lr = lr
            self.epochs = epochs
            self.batch_size = batch_size
            self.weight_decay = weight_decay
            self.geometric_criterion = nn.MSELoss(reduction='sum')
            if regression:   
                self.classification_criterion = nn.MSELoss()
            else:
                # self.classification_criterion = nn.BCELoss()
                self.classification_criterion = nn.CrossEntropyLoss()
            
            self.lam = lam
            
            self.z = None       
            if Embedding is not False:
                self.z_dim= Embedding.shape[1]
                self.Embedding = Embedding
                Embedding = scipy.stats.zscore(self.Embedding)
                self.z = torch.from_numpy(Embedding).float().to(device)
                
            
        def fit(self, X, epochs=None, epoch_offset=0):
            if epochs is None:
                epochs = self.epochs     
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            # print(device)
            # Reproducibility
            # torch.manual_seed(self.random_state)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            
            idx2 = torch.where(~torch.isnan(X.targets))[0]

            
            self.z_labels = self.z[idx2,:]
            self.data_labels = X.data[idx2,:]
            self.y_labels = X.targets[idx2].squeeze()
            self.n_classes = len(torch.unique(self.y_labels))
            
            
            if self.torch_module is None:
                input_size = X[0][0].shape[0]
                self.torch_module = SemiSupModuleAllR(input_dim = input_size,
                                                   z_dim = self.z_dim,
                                                   n_classes = self.n_classes)
    
    
            self.optimizer = torch.optim.Adam(self.torch_module.parameters(),
                                              lr=self.lr,
                                              weight_decay=self.weight_decay)
            # Train AE
            self.torch_module.to(device)
            self.torch_module.train()
    
            self.loader = self.get_loader(X)
    
            for epoch in range(epochs):
                print(f'            Epoch {epoch + epoch_offset}...')
                for batch in self.loader:
                    self.optimizer.zero_grad()
                    self.train_body(batch)
                    self.optimizer.step()
                
                self.end_epoch(epoch)
        
        def get_loader(self, X):
            return torch.utils.data.DataLoader(X, batch_size=self.batch_size, shuffle=True)
        
        def train_body(self, batch):
            data, y, idx = batch
            # data = torch.cat((data1, self.data_labels))
            # y = torch.cat((y1, self.y_labels))
            data = data.to(device)
            y_pred, r1, r2, r3, r4 = self.torch_module(data)
            self.apply_loss(y_pred, r1, r2, r3, r4, y, idx)
        
        def apply_loss(self, y_pred, r1, r2, r3, r4, y, idx):
            E = self.z[idx]
            loss_g = self.geometric_criterion(E, r1) +  self.geometric_criterion(E, r2) +  self.geometric_criterion(E, r3) + 0*self.geometric_criterion(E, r4) 
            idx2 = torch.where(~torch.isnan(y))[0]
            y = y.to(device).long()
            loss_C = self.classification_criterion(y_pred[idx2], y[idx2])
            loss = self.lam*loss_g + loss_C
            loss.backward()
        
        def end_epoch(self, epoch):
            pass
        
