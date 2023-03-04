
############# < 정리한 실행 부분 > #################


import numpy as np

x1 = np.array([[1,2], [3,4]])
x2 = np.array([[[1,2,3]]])
x3 = np.array([[[1,2,3], [4,5,6]]])
x4 = np.array([[1], [2], [3]])
x5 = np.array([[[1]], [[2]], [[3]]])
x6 = np.array([[[1,2], [3,4]], [[5,6], [7,8]]])
x7 = np.array([[[1,2]], [[3,4]], [[5,6]], [[7,8]]])
x8 = np.array([[[[1]], [[2]], [[3]]], [[[1]], [[2]], [[3]]]])

print(x1.shape) 
print(x2.shape) 
print(x3.shape) 
print(x4.shape) 
print(x5.shape) 
print(x6.shape) 
print(x7.shape) 
print(x8.shape) 


################ < 작업 결과 > #####################

#  (2, 2)
#  (1, 1, 3)
#  (1, 2, 3)
#  (3, 1)
#  (3, 1, 1)
#  (2, 2, 2)
#  (4, 1, 2)
#  (2, 3, 1, 1)
#

################# < 수업내용 > ######################


#  import numpy as np

#  x1 = np.array([[1,2], [3,4]])
#  x2 = np.array([[[1,2,3]]])
#  x3 = np.array([[[1,2,3], [4,5,6]]])
#  x4 = np.array([[1], [2], [3]])
#  x5 = np.array([[[1]], [[2]], [[3]]])
#  x6 = np.array([[[1,2], [3,4]], [[5,6], [7,8]]])
#  x7 = np.array([[[1,2]], [[3,4]], [[5,6]], [[7,8]]])
#  x8 = np.array([[[[1]], [[2]], [[3]]], [[[1]], [[2]], [[3]]]])

#  print(x1.shape) # (2, 2) ---> ( 2행 2열 ) matrix ---> ( 2 x 2 ) matrix
#  print(x2.shape) # (1, 1, 3) ---> ( 1 x 1 x 3 ) matrix
#  print(x3.shape) # (1, 2, 3)
#  print(x4.shape) # (3, 1)
#  print(x5.shape) # (3, 1, 1)
#  print(x6.shape) # (2, 2, 2)
#  print(x7.shape) # (4, 1, 2)
#  print(x8.shape) # (2, 3, 1, 1) ---> ( 2 x 3 x 1 x 1 ) matrix
#
#