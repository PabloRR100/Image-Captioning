import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)                       # Created the pre-trained model imported from models.resnet50
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)   # Attach the embed (dense) layer to map FV to embed size
        self.norm = nn.BatchNorm1d(embed_size, momentum=0.01)       # Attach Batch Normalization Layer to increase Regularization

    def forward(self, images):
        with torch.no_grad():
            features = self.resnet(images)              # Forward pass throuth the entire ResNEt
        features = features.view(features.size(0), -1)  # Flatten the resulting feature vector
        features = self.embed(features)                 # Forward pass feature vector through linear to adjust to embed_size
        features = self.norm(features)                  # Batch Norm Layer
        # features = self.norm(features)                  # Normalize the current batch of features
        return features

        
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        
        self.embed = nn.Embedding(vocab_size, embed_size)                                # Map input (vocab) size to embed size
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first = True)  # LSTM
        self.linear = nn.Linear(hidden_size, vocab_size)                              # Map back from hidden size to vocab size
        self.dropout = nn.Dropout(0.3)
            
    def forward(self, features, captions):
        ''' Decode image feature vectors and generates captions '''  
        
        '''
        #### QUESTION:
        Shouldnt' captions be now one-hot-encoded (or in a pre-process step) to reshape to (batch_size * vocab_size) ?
        Since now they are going through the Embed layer that expects vocab_size dim vector to reshape it into embed_size
        '''
        captions = captions[:,:-1]                                      # Remove <end> token
        embeddings = self.embed(captions)                               # Map (FORMATED) captions to embed size (Dense layer)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)  # Concatenate (FLATTENED) FV with (EMBEDDED) captions
        out, hidden = self.lstm(embeddings)                             # Forward pass LSTM 
        outputs = self.linear(out)                                      # Map output to vocab_size (the scores)      
        return outputs                        
        # From notes: ==> outputs should be a PyTorch tensor with size [batch_size, captions.shape[1], vocab_size] (which is not?)
        
        '''
        # Attention mechanism would have been here, or we need to overwrite torch.nn.LSTM ???
        1 - We flatten the output volume of resnet into the Context Vector
        2 - In the LSTM, for each time step:
            2.1 - Compute the scores a(s) = align(current_hidden_state, context_vector)
            2.2 - Calculate the annotation by multiplying the 'softmaxed' weights by the context_vector
            2.3 - Concatenate the annotation with the current hidden_state vector, and multiply it by Wc (will be trained)
            2.4 - Apply tanh to that to obtain the output of that hidden_state
        '''
        
    def sample(self, inputs, s=None, max_len=20):
        " Accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        ids = []
        for i in range(max_len):
            
            h, s = self.lstm(inputs, s)         # Input the features into the lstm     -- [h]: (batch_size, 1, hidden_size)
            outputs = self.linear(h.squeeze(1)) # Map from hidden states to embed size -- [o]: (batch_size, 1, embed_size)
            _, pred = outputs.max(1)            # Apply Softmax and kee the max score  -- [p]: (batch_size) 
            ids.append(pred)                    # Append new predicted id
            
            inputs = self.embed(pred)           # Incorporate output to next input -- [i]: (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)        # Adding single-value dimension    -- [i]: (batch_size, 1, embed_size)
            
        ids = torch.stack(ids, 1)               # [Samples ids]: (batch_size, max_len)
        ids = ids.cpu().numpy().squeeze().tolist()
        return ids
            
            
            
        
        