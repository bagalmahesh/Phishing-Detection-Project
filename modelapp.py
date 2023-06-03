#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flask import Flask, request, jsonify, Response
import torch
import torch.nn as nn
from flask_cors import CORS
from model import GRUModel


# In[2]:


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})


# In[3]:


# Load the model
input_size = 100
hidden_size = 64
output_size = 2
num_layers = 2

# Load the model and define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GRUModel(input_size,hidden_size,output_size,num_layers)  # Modify this line to load your model
model.load_state_dict(torch.load('model.pth', map_location=device))
model = model.to(device)
model.eval()


# In[4]:


@app.route('/epredict', methods=['POST'])
def predict():
    url = request.json['url']
    feature = convert_url_to_feature(url)  # Modify this line to extract features from the URL
    feature = feature.unsqueeze(0)
    feature = feature.float().to(device)

    with torch.no_grad():
        output = model(feature, device)
        probabilities = nn.functional.softmax(output, dim=1)[0]
        predicted_class = torch.argmax(probabilities).item()

    result = "Phishing" if predicted_class == 1 else "Legitimate"
    # Set CORS headers
    response = jsonify({'result': result})
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Methods', 'POST')  # Set the allowed methods if needed
    response.headers.add('Access-Control-Allow-Private-Network', 'true')  # Add this header to allow private network access

    
    return response


# In[5]:


def convert_url_to_feature(url):
    char_dict = {
        '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
        'a': 10, 'b': 11, 'c': 12, 'd': 13, 'e': 14, 'f': 15, 'g': 16, 'h': 17, 'i': 18, 'j': 19,
        'k': 20, 'l': 21, 'm': 22, 'n': 23, 'o': 24, 'p': 25, 'q': 26, 'r': 27, 's': 28, 't': 29,
        'u': 30, 'v': 31, 'w': 32, 'x': 33, 'y': 34, 'z': 35, 'A': 36, 'B': 37, 'C': 38, 'D': 39,
        'E': 40, 'F': 41, 'G': 42, 'H': 43, 'I': 44, 'J': 45, 'K': 46, 'L': 47, 'M': 48, 'N': 49,
        'O': 50, 'P': 51, 'Q': 52, 'R': 53, 'S': 54, 'T': 55, 'U': 56, 'V': 57, 'W': 58, 'X': 59,
        'Y': 60, 'Z': 61, '-': 62, ',': 63, ';': 64, '.': 65, '!': 66, '?': 67, ':': 68, "'": 69,
        '"': 70, '/': 71, '\\': 72, '|': 73, '_': 74, '@': 75, '#': 76, '$': 77, '%': 78, '^': 79,
        '&': 80, '*': 81, '~': 82, '`': 83, '+': 84, '-': 85, '=': 86, '<': 87, '>': 88, '(': 89,
        ')': 90, '[': 91, ']': 92, '{': 93, '}': 94, '\t': 95, '\n': 96, '\x0b': 97, '\x0c': 98
    }
    
    feature = torch.zeros(200, 100)
    for i, char in enumerate(url):
        if i >= 200:
            break
        if char in char_dict:
            feature[i, char_dict[char]] = 1
    
    return feature


# In[6]:


if __name__ == '__main__':
    app.run()


# In[ ]:




