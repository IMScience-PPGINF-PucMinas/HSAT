# -*- coding: utf-8 -*-
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import torch
from PIL import Image
import torchvision.transforms as transforms
import json
import warnings
import networkx as nx
import imageio
# %matplotlib inline
import matplotlib.pyplot as plt
import higra as hg
import os
import http.client
from joblib import Parallel, delayed
from multiprocessing import Pool

try:
    from utils import * # imshow, locate_resource, get_sed_model_file
except: # we are probably running from the cloud, try to fetch utils functions from URL
    import urllib.request as request; exec(
      request.urlopen('https://github.com/higra/Higra-Notebooks/raw/master/utils.py'
      ).read(), globals())

warnings.filterwarnings('ignore')
# %matplotlib inline

device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")
print('Using ' + str(device))

def ext_resnet(file_dir, net, utils):
  uris = [
      file_dir
  ]

  batch = torch.cat(
      [utils.prepare_input_from_uri(uri) for uri in uris]
  ).to(device)

  with torch.no_grad():
      output = torch.nn.functional.softmax(net(batch), dim=1)
  return output

def cos_similarity(frame1, frame2): #images to compute a similarity with cosine_similarity metric
  cos_sim =100 * cosine_similarity(
    frame1, frame2, dense_output=True)[0][0]
  return float(cos_sim)

def calculateScore(matches, keypoint1, keypoint2):

  len1 = len(keypoint1)
  len2 = len(keypoint2)
  lenm = len(matches)

  if (len1 == len2) and (len1 == lenm):
    return 0
  return round(100 * (lenm/min(len1, len2)), 2)

def calculateMatches(des1,des2):

  distance = 0.75

  FLANN_INDEX_KDTREE = 1
  index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
  search_params = dict(checks = 50)
  flann = cv.FlannBasedMatcher(index_params, search_params)
  
  if(des1 is not None and len(des1)>2 and des2 is not None and len(des2)>2):
      matches =  flann.knnMatch(des1, des2, k = 2)
      topResults1 = [[0, 0] for i in range(len(matches))]

      for i,(m,n) in enumerate(matches):
          if m.distance < distance * n.distance:
              topResults1[i]=[1, 0]

  if(des1 is not None and len(des1) > 2 and des2 is not None and len(des2) > 2):

      matches =  flann.knnMatch(des2, des1, k = 2)
      topResults2 = [[0,0] for i in range(len(matches))]

      for i,(m, n) in enumerate(matches):
          if m.distance < distance * n.distance:
              topResults2[i]=[1, 0]        

  topResults = []

  for  item1, item2 in zip(topResults1, topResults2):
      if (item1[0] == item2[0] and item1[1] == item2[1]):
          topResults.append(item1)

  return topResults

def mySift(img1, img2):
  sift = cv.SIFT_create()
  dist = 0.75

  kp1, des1 = sift.detectAndCompute(img1, None)
  kp2, des2 = sift.detectAndCompute(img2, None)

  bf = cv.BFMatcher()

  score = 100

  if(des1 is not None and len(des1) > 2 and des2 is not None and len(des2) > 2):

      matches = bf.knnMatch(des1, des2, k = 2) 
      matchesMask  = [[0, 0] for i in range(len(matches))]

      for i,(m, n) in enumerate(matches):
          if m.distance < dist * n.distance:
              matchesMask[i]=[1, 0]  

      matches = calculateMatches(des1, des2)
      score = calculateScore(matches, kp1, kp2)
  return score

def calc_init(i, delta_t, frame_len):
  if(i < delta_t):
    return 0
  elif((i + delta_t) > frame_len):
    return i
  else:
    return i - delta_t

def calc_end(i, delta_t, frame_len):
  if(i < delta_t):
    return delta_t
  elif((i + delta_t) > frame_len):
    return frame_len
  else:
    return i + delta_t

def saveData(file, v1, v2, weight):
  f = open(file, "a")
  if(v2==' '):
    data = "{}\n".format(v1)
  elif(weight == ".jpg"):
    data = "{}{}\n".format(v1, weight)
  else:
    data = "{}, {}, {:.2f}\n".format(v1, v2, weight)
  f.write(data)
  f.close()

def plotGraph(PG, not_weighted):
  elarge = [(u, v) for (u, v, d) in PG.edges(data=True)]
  pos = nx.spring_layout(PG, seed=7)  # positions for all nodes - seed for reproducibility
  nx.draw_networkx_edges(
      PG, pos, edgelist = elarge, width=1, alpha=1
  )

  nx.draw_networkx_labels(PG, pos)

  # nodes
  nx.draw_networkx_nodes(PG, pos)
  if(not_weighted):
    ax = plt.gca()
    plt.show()
  
  else:
    # edge weight labels
    edge_labels = nx.get_edge_attributes(PG, "weight")
    nx.draw_networkx_edge_labels(PG, pos, edge_labels)
    ax = plt.gca()
    plt.show()

def loadFrames(video_file, sift, delta_t, output_graph_file, net, utils):
  if(os.path.exists(video_file)):
    frame_list = os.listdir(video_file)
    frame_list.sort() # to garanted the time order
    frame_len = len(os.listdir(video_file))
    features_list = [
        ext_resnet(video_file + frames, net, utils).numpy()
        for frames in frame_list] #List of frame features
    feat_list_len = len(features_list)

    for vertex1 in range(feat_list_len):
      processes = [saveData(
          output_graph_file, 
              "{}, {}, {:.2f}".format(vertex1, vertex2, cos_similarity(
              features_list[vertex1], features_list[vertex2]))
              , ' ', ' ') 
              for vertex2 in range(calc_init(vertex1, delta_t, feat_list_len),
                                   calc_end(vertex1, delta_t, feat_list_len))]

def gen_mst(input_graph_file, input_mst):
  VG = readGraphFile(input_graph_file, ' ', cut_graph = False, cut_number = 0)
  T = nx.minimum_spanning_tree(VG)
  for h in sorted(T.edges(data=True)):
    saveData(input_mst, h[0], h[1], h[2]["weight"])
  return T

def computeHierarchy(input_g, isbinary, input_higra):
  leaf_list = []
  graph = hg.UndirectedGraph()       #convert the scikit image rag to higra unidrect graph
  graph.add_vertices(max(input_g._node)+1)   #creating the nodes (scikit image RAG starts from 1)
  edge_list = list(input_g.edges())                   #ScikitRAG edges

  for i in range (len(edge_list)):
      graph.add_edge(edge_list[i][0], edge_list[i][1]) #Adding the nodes to higra graph
  edge_weights = np.empty(shape=len(edge_list))
  sources, targets = graph.edge_list()

  for i in range (len(sources)):    
    edge_weights[i] = int(input_g.adj[sources[i]][targets[i]]["weight"])
  nb_tree, nb_altitudes = hg.watershed_hierarchy_by_area(graph, edge_weights)
  
  if(isbinary):
    tree, node_map = hg.tree_2_binary_tree(nb_tree)
    altitudes = nb_altitudes[node_map]
  else:
    tree = nb_tree
    altitudes = nb_altitudes

  for n in tree.leaves_to_root_iterator():
    leaf = -2 # It's cod is used for the node that is not a leaf
    if(tree.is_leaf(n)):
      leaf = -1 # It's cod is used for the node that is a leaf
      leaf_list.append(n)
    saveData(input_higra, n, tree.parent(n), leaf)
  return(leaf_list)

def loadNet(type, pretrained):
  if(type is "resnet50"):
    net = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub',
      'nvidia_resnet50', pretrained=pretrained)
    utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub',
      'nvidia_convnets_processing_utils')
  elif(type is "resnet101"):
    net = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub',
      'nvidia_resnet101', pretrained=pretrained)
    utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub',
      'nvidia_convnets_processing_utils')
  return net, utils

def readGraphFile(file, cut_graph_file, cut_graph, cut_number):
  RG = nx.Graph()
  with open(file) as f:
    lines = f.readlines()

  cut = len(lines)
  if(cut_graph):
    cut -= (cut_number +1)

  for line in lines:
    w = float(line.split(", ")[2]) # weight
    v1 =int(line.split(", ")[0]) # node 1
    v2 =int(line.split(", ")[1]) # node 2
    if(cut != 0):
      RG.add_edge(v1, v2, weight = w) # include two node and your weight
      cut -=1
      if(cut_graph):
        saveData(cut_graph_file, v1, v2, w)

    else:
      if(w == -1):
        v1 =int(line.split(", ")[0]) # node 1
        v2 =int(line.split(", ")[1]) # node 2
        if(v1 <= v2):
          if(not (v2 in cutlist)):
            RG.add_node(v1)
            saveData(cut_graph_file, v1, ' ', ' ')
            cutlist.append(v2)
            cut -=1
        else:
          if(not (v1 in cutlist)):
            RG.add_node(v2)
            saveData(cut_graph_file, v2, ' ', ' ')
            cutlist.append(v1)
            cut -=1
  return RG

def selectKeyFrame(graph, key_frame, leaflist):
  S = [graph.subgraph(c).copy() for c in nx.connected_components(graph)]
  for c in range(len(S)):
    plotGraph(S[c], True)
    central_node = len(S[c].nodes)
    comp_leaf_list = []
    for i in range(central_node):
      if(list(S[c])[i] in leaflist):
        comp_leaf_list.append(list(S[c])[i])
    cn = int(len(comp_leaf_list)/2)
    kf = str(comp_leaf_list[cn]).zfill(6)
    saveData(key_frame, kf, '  ', '.jpg')

def main(video_file, rate, time, net, utils):
  input_graph_file = video_file + 'graph.txt'
  input_mst = video_file + 'mst.txt'
  input_higra = video_file + 'higra.txt'
  key_frame = video_file + 'keyframe.txt'
  cut_graph_file = video_file + 'cut_graph.txt'
  delta_t = rate * time # how many frame to get for the time, for example with time equal to 4 is equivalant to  4 seconds of video and for rate equal to 2 is equal to 2 frames for each second, in this case the firt cut is equal to 8 frames
  sift = False # if True, the features extraction to compute differences are SIFT; else use the Resnet50.

  if(not os.path.exists(input_graph_file)):
    print("Create Graph for the File {}".format(video_file))
    loadFrames(video_file, sift, delta_t, input_graph_file, net, utils) # Load the frame list and create a graph for the video
    tree = gen_mst(input_graph_file, input_mst) # generate the minimum spanning tree
    isbinary = True # To compute a binary hierarchy
    leaflist = computeHierarchy(tree, isbinary, input_higra) # Create the hierarchy based on the minimum spanning tree and return the leaves of the new hierarchy
    cuted_graph = readGraphFile(input_higra, cut_graph_file, cut_graph = True, cut_number = 10) # Create a new graph based on the hierarchy and the level cut
    selectKeyFrame(cuted_graph, key_frame, leaflist) # With the cuted graph, create a keyframe to represent each component or segment of video

if __name__ == "__main__":
  import sys
  dataset = sys.argv[1]
  rate = int(sys.argv[2])
  time = int(sys.argv[3])

  net_type = "resnet50"
  pretrained = True
  net, utils = loadNet(net_type, pretrained)

  if(os.path.exists(dataset)):
    video_list = os.listdir(dataset)
    video_list.sort() # to garantee order
    for video in video_list:
      main("{}/{}/".format(dataset, video), rate, time, net, utils)
  print("Done")
    