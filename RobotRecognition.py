import numpy as np

model_filename='IIWA_model.obj'
scene_filename='Selection7.obj'

floor_treshold=-0.8
ceiling_treshold=1.5
model_filter=10
scene_filter=1
model_share=0.5
area_coefficient_1=0.95
area_coefficient_2=5
model_layers=100

# Считывание облака точек из формата obj
def read_point_cloud(filename):
    import pandas as pd
    data = pd.read_csv(filename, sep=' ', header=None)
    data = np.asarray(data)
    data = data[data[:, 0] == 'v']
    point_cloud = data[:, 1:4]
    point_cloud = np.asarray(point_cloud, float)
    return point_cloud

# Фильтрация облака точек, также функция расставляет координаты точек в порядке (x y z)
def filtering(A,filter):
    filtering_A=[]
    for i in range (int(A.shape[0]/filter)):
        filtering_A.append([A[filter*i,0].copy(),A[filter*i,2].copy(),A[filter*i,1].copy()])
    filtering_A=np.asarray(filtering_A)
    return filtering_A

# Рассчет площади объекта по координатам. Алгоритм основан на формуле Гаусса для площади
def compute_area(A):
    from math import fabs
    S=0
    for i in range(A.shape[0]-1):
        S=S+1/2*fabs(A[i,0]*A[i+1,1]-A[i+1,0]*A[i,1])
    S=S+1/2*fabs(A[A.shape[0]-1,0]*A[0,1]-A[0,0]*A[A.shape[0]-1,1])
    return S

#Функция, определяющая с какой стороны от вектора AB находится точка С (положительное значение - слева, отрицательное - справа)
#на вход подаются координаты трех точек на плоскости
def place_point_position(A,B,C):
    return (B[0]-A[0])*(C[1]-B[1])-(B[1]-A[1])*(C[0]-B[0])

#Функция, определяющая с какой стороны от вектора AB находится точка С (положительное значение - слева, отрицательное - справа)
#на вход подается координаты трех точек на плоскости
def jarvismarch(A):
    P=np.argsort(A[:,0])
    H=[P[0]]
    P=np.hstack([P[1:],P[0]])
    while True:
        right=0
        for i in range(1,P.shape[0]):
            if place_point_position(A[H[-1]],A[P[right]],A[P[i]])<0:
                right=i
        if P[right]==H[0]:
            break
        else:
            H.append(P[right])
            P=np.hstack([P[0:right],P[right+1:]])
    return A[H],H

#Поиск минимальной выпуклой оболочки алгоритмом Джарвиса для точек на плоскости
#на вход подается 2D-массив точек
def rationing_points(A):
    X=A[:,0:2].copy()
    X[:,0]=X[:,0]-np.mean(X[:,0])
    X[:,1]=X[:,1]-np.mean(X[:,1])
    return X

# Задает начальное положение модели в сцене с учетом направления кластеризованного объекта
def trans_init(A,B,Ztreshold):
    from math import sqrt
    indecs=np.where(A[:,2]>np.max(A[:,2])-Ztreshold)
    A_vector=A[np.where(A[:,2]==np.max(A[:,2]))][0,0:2]
    A_vector=np.vstack([A_vector,[np.mean(A[indecs,0]),np.mean(A[indecs,1])]])
    indecs=np.where(B[:,2]>np.max(A[:,2])-Ztreshold)
    B_vector = B[np.where(B[:, 2] == np.max(B[:, 2]))][0, 0:2]
    B_vector = np.vstack([B_vector, [np.mean(B[indecs, 0]), np.mean(B[indecs, 1])]])
    cos_alpha=((A_vector[0][0]-A_vector[1][0])*(A_vector[0][1]-A_vector[1][1])+(B_vector[0][0]-B_vector[1][0])*(B_vector[0][1]-B_vector[1][1]))/(sqrt((A_vector[0][0]-A_vector[1][0])**2+(A_vector[0][1]-A_vector[1][1])**2)*sqrt((B_vector[0][0]-B_vector[1][0])**2+(B_vector[0][1]-B_vector[1][1])**2))
    sin_alpha=sqrt(1-cos_alpha**2)
    trans_init = np.asarray(
                [[cos_alpha, -sin_alpha, 0.0,  -(B_vector[0][0]-A_vector[0][0])],
                [sin_alpha, cos_alpha, 0.0,  -(B_vector[0][1]-A_vector[0][1])],
                [0.0, 0.0,  1.0, -(np.max(B[:,2])-np.max(A[:,2]))],
                [0.0, 0.0, 0.0, 1.0]])
    return trans_init

# Функция на выходе дает матрицу трансформации
def best_fit_transform(A, B):
    m = A.shape[1]
    # Рассчет центроидов для трансляции
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B
    # Рассчет матрицы поворота
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    # Получение матрицы трансформации
    if np.linalg.det(R) < 0:
        Vt[m-1,:] *= -1
        R = np.dot(Vt.T, U.T)
    t = centroid_B.T - np.dot(R, centroid_A.T)
    T = np.identity(m+1)
    T[:m, :m] = R
    T[:m, m] = t
    return T, R, t

# Получение матрицы трансформации
def nearest_neighbor(src, dst):
    from sklearn.neighbors import NearestNeighbors
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()

# Поиск ближайших соседей
def icp(A, B, init_pose=None, max_iterations=100):
    assert A.shape[1] == B.shape[1]
    m = A.shape[1]
    src = np.ones((m+1,A.shape[0]))
    dst = np.ones((m+1,B.shape[0]))
    src[:m,:] = np.copy(A.T)
    dst[:m,:] = np.copy(B.T)
    if init_pose is not None:
        src = np.dot(init_pose, src)
    prev_error = np.max(A)
    n=0
    for i in range(max_iterations):
        # Поиск ближайших соседей между текущими положениями сцены и модели
        distances, indices = nearest_neighbor(src[:m,:].T, dst[:m,:].T)
        # Рассчет трансформации между текущим положением модели и ближайшими точками сцены
        T,_,_ = best_fit_transform(src[:m,:].T, dst[:m,indices].T)
        src = np.dot(T, src)
        mean_error = np.mean(distances)
        if mean_error<prev_error:
            prev_error=mean_error
            best_src=src
            n=i
    # Рассчет финальной трансформации
    T,_,_ = best_fit_transform(A, best_src[:m,:].T)
    return T, prev_error,n

# Для визуализации
def animate(a1,a2,a3,a4,name):
    import matplotlib.pyplot as plt
    plt.plot(a1,a2,'o', color='Black',label='Scene PC')
    plt.plot(a3,a4,'o',color='Red',label='Object boundaries')
    plt.title(name)
    plt.show()

# Считывание облака точек модели и сцены
model=read_point_cloud(model_filename)
scene=read_point_cloud(scene_filename)

# Удаление пола и элементов потолка по трешолду
scene=scene[scene[:,1]>floor_treshold]
scene=scene[scene[:,1]<ceiling_treshold]

# Фильтрация модели и сцены, уменьшение облака точек в model_filter раз
filtering_model=filtering(model,model_filter)
filtering_scene=filtering(scene,scene_filter)

# Кластеризация объектов в облаке точек сцены
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

X = StandardScaler().fit_transform(filtering_scene[:,0:2])

db = DBSCAN(eps=0.2, min_samples=10).fit(X)# Расчет DBSCAN кластеров
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

n_clusters = len(set(labels)) - (1 if -1 in labels else 0) # Количество кластеров в сцене (label -1 означает шум)

X=rationing_points(filtering_model[:,0:2]) # Нормировка точек минимальной выпуклой оболочки на нулевое положение для расчета площади занимаемой поверхности
X,_=jarvismarch(X)
model_area=compute_area(X) # Расчет площади модели (вид сверху)

# Подбор кластеризованных объектов по площади занимаемой ими поверхности по отношению к площади модели
objects=[]
for i in range(n_clusters):
    auxiliary_cloud =filtering_scene[labels==i]
    X=rationing_points(auxiliary_cloud[:,0:2])
    X, _ = jarvismarch(X)
    auxiliary_area=compute_area(X)
    if (auxiliary_area>area_coefficient_1*model_area and auxiliary_area<area_coefficient_2*model_area):
        objects.append(auxiliary_cloud)

# Построение минимальной выпуклой оболочки модели для удаления внутренних точек
step=(np.max(filtering_model[:,0])-np.min(filtering_model[:,0]))/model_layers
shell_model=filtering_model[0:2,:]
for i in range(model_layers-1):
    auxiliary_cloud=filtering_model[filtering_model[:,0]>np.min(filtering_model[:,0])+i*step]
    auxiliary_cloud=auxiliary_cloud[auxiliary_cloud[:,0]<np.min(filtering_model[:,0])+(i+1)*step]
    if auxiliary_cloud.shape[0]==0 or auxiliary_cloud.shape[1]==0:
        continue
    _,ind=jarvismarch(auxiliary_cloud[:,1:3])
    shell_model=np.concatenate([shell_model,auxiliary_cloud[ind]])
shell_model=shell_model[2:,:]

# Алгоритм ICP для каждого из подходящих кластеризованных объектов
model_treshold=(np.max(filtering_model[:,2])-np.min(filtering_model[:,2]))*model_share
best_T,best_error,_=icp(filtering_model,objects[0])
n=0
for i in range(len(objects)):
    init_pose = trans_init(objects[i], filtering_model, model_treshold)
    T,error,_=icp(shell_model,objects[i],init_pose=init_pose)
    if error<best_error:
        best_error=error
        best_T=T
        n=i
    src = np.ones((filtering_model.shape[1] + 1, filtering_model.shape[0]))
    src[:filtering_model.shape[1], :] = np.copy(filtering_model.T)
    src = np.dot(T, src)
    src = src.T
    dst = objects[i]

    animate(dst[:, 0], dst[:, 1], src[:, 0], src[:, 1], 'final transform according by clustering object '+str(i))
    animate(dst[:, 0], dst[:, 2], src[:, 0], src[:, 2], 'final transform according by clustering object '+str(i))
    animate(dst[:, 1], dst[:, 2], src[:, 1], src[:, 2], 'final transform according by clustering object '+str(i))

print('Homogeneous transformation for this model in scene point cloud is:')
print(best_T)

# # Для визуализации финальной трансформации
# src = np.ones((filtering_model.shape[1] + 1, filtering_model.shape[0]))
# src[:filtering_model.shape[1], :] = np.copy(filtering_model.T)
# src=np.dot(best_T,src)
# src=src.T
# dst=objects[n]
#
# animate(dst[:,0],dst[:,1],src[:,0],src[:,1],'final transform according by clustering object')
# animate(dst[:,0],dst[:,2],src[:,0],src[:,2],'final transform according by clustering object')
# animate(dst[:,1],dst[:,2],src[:,1],src[:,2],'final transform according by clustering object')
#
# animate(filtering_scene[:,0],filtering_scene[:,1],src[:,0],src[:,1],'final transform according by scene point cloud')
# animate(filtering_scene[:,0],filtering_scene[:,2],src[:,0],src[:,2],'final transform according by scene point cloud')
# animate(filtering_scene[:,1],filtering_scene[:,2],src[:,1],src[:,2],'final transform according by scene point cloud')