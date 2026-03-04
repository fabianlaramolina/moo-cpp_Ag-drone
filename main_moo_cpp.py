import numpy as np
import math
import os
from matplotlib.patches import Polygon
from shapely.geometry import LineString, Polygon, Point
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2

from pymoo.algorithms.moo.moead import MOEAD
from pymoo.util.ref_dirs import get_reference_directions

from pymoo.algorithms.moo.spea2 import SPEA2

from pymoo.optimize import minimize
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.visualization.scatter import Scatter
import matplotlib.pyplot as plt
import geopandas as gpd
import contextily as ctx
from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD

import time



def polygon_circular(raio,n_vertices):
    # Parâmetros do círculo
    #raio = 100
    centro = np.array([0, 0])
    #n_vertices = 36
    
    # Ângulos igualmente espaçados
    angulos = np.linspace(0, 2 * np.pi, n_vertices, endpoint=False)
    
    # Coordenadas dos vértices
    x = centro[0] + raio * np.cos(angulos)
    y = centro[1] + raio * np.sin(angulos)
    
    # Combina em array de vértices 2D
    vertices_circulares = np.column_stack((x, y))
    
    # Fecha o polígono adicionando o primeiro vértice ao final
    vertices_circulares = np.vstack([vertices_circulares, vertices_circulares[0]])
   
    return vertices_circulares    
      
        




print("1. Poligono 1 (nonegono):")
print("2. Poligono 2: (pentagono)")
print("3. Poligono 3: (retangulo)")
print("4. Poligono 4: (circulo)")
print("5. Mapa - Caso Estudo:")

try:
    choice_pol = int(input("Enter your choice (1-3): "))
except ValueError:
    choice = 0 

if choice_pol==1:
    FR=.75
    vertices = np.array([[-40*FR, 75*FR], [-75*FR, 0*FR], [-40*FR, -75*FR],[0,-85*FR], [40*FR, -75*FR], [100*FR, -60*FR], [125*FR, 00*FR], [40*FR, 75*FR],[0, 85*FR], [-40*FR, 75*FR]]) # Poligon 7:9v
    home_pos = np.array([-60*FR, -50*FR]) 
if choice_pol==2:
    FR=.45  
    vertices = np.array([[0, 150*FR], [-150*FR, 0], [-80*FR, -150*FR], [80*FR, -150*FR], [150*FR, 00], [0, 150*FR]]) # Poligon 3:5v
    home_pos=np.array([-110*FR, 60*FR])
if choice_pol==3:
    FR=.45  
    vertices = np.array([[-100*FR, 150*FR], [-100*FR, -150*FR], [100*FR, -150*FR], [100*FR, 150*FR],[-100*FR, 150*FR]]) # Poligon 2: 4v
    home_pos=np.array([-110*FR, 0*FR])
if choice_pol==4:
    FR=.45
    raio=150*FR
    n_vertices=36
    vertices = polygon_circular(raio,n_vertices)  # Poligon 8
    home_pos=np.array([160*FR, 0*FR])
        
if choice_pol==5:
    # Caminho até o arquivo .shp
    caminho_shp = '~/MOO_CPP_python/area_UFTM.shp'
    # Lê o shapefile como GeoDataFrame
    gdf = gpd.read_file(caminho_shp)
    # Reprojetar para EPSG:3857 (necessário para o mapa base do contextily)
    gdf_proj = gdf.to_crs(epsg=3857)
    # Plotar area
    ax = gdf_proj.plot(figsize=(7.5, 5), alpha=0.0, edgecolor='red')
    
    # Extrair o polígono da primeira linha
    geom = gdf_proj.geometry.iloc[0]
    
    # Se for multipolígono, pega o polígono externo
    if geom.geom_type == "MultiPolygon":
        geom = list(geom)[0]

    # Extrai as coordenadas do contorno exterior
    vertices = np.array(geom.exterior.coords)
    
    print("Vértices extraídos:")
    print(vertices)
    
    # Adicionar mapa base (OpenStreetMap por padrão)
    #ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
    ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery,zoom=18) #plotar Mapa # valores típicos: 15 (cidade), 17–19 (bairro/terreno)
    plt.plot(vertices[:, 0], vertices[:, 1], 'k-')  # 'b--' é azul tracejado
    
    home_pos = np.array([-5339494.74947602, -2238364.65294439-10])
    plt.plot(home_pos[0], home_pos[1], marker='^', color='orange')
    
    
    x_pos = gdf_proj.total_bounds[0] + 0.05 * (gdf_proj.total_bounds[2] - gdf_proj.total_bounds[0])
    y_pos = gdf_proj.total_bounds[1] + 0.95 * (gdf_proj.total_bounds[3] - gdf_proj.total_bounds[1])
    ax.annotate('N', xy=(x_pos, y_pos), xytext=(x_pos, y_pos + 0.05),
        arrowprops=dict(facecolor='red', width=4, headwidth=20),
        ha='center', fontsize=22, fontweight='bold',color='red')
    
    plt.show()
    
   
    
    

# Separar coordenadas x e y
x = vertices[:, 0]
y = vertices[:, 1]
       
# Fórmula de Shoelace
area = 0.5 * np.abs(np.dot(x[:-1], y[1:]) - np.dot(x[1:], y[:-1]))
# Converter para hectares (1 ha = 10,000 m²)
area_ha = area / 10_000
#print(f"Área = {area:.2f} m²")
print(f"Área = {area_ha:.4f} hectares")

omega=90*(math.pi/180) #1 #Velocidade Angular do Drone (rad/s)  [57 graus/s]  


# ================================================================
# Definição do problema
# ================================================================
# class MyProblem(ElementwiseProblem):
# 
#     def __init__(self):
#         super().__init__(
#             n_var=2,              # número de variáveis de decisão
#             n_obj=2,              # número de objetivos
#             n_constr=0,           # número de restrições
#             xl=np.array([-5, -5]),  # limites inferiores
#             xu=np.array([ 5,  5])   # limites superiores
#         )
# 
#     def _evaluate(self, x, out, *args, **kwargs):
#         # Funções objetivo
#         f1 = x[0]**2 + x[1]**2
#         f2 = (x[0]-1)**2 + (x[1]-1)**2
#         out["F"] = [f1, f2]
        

# ================================================================
# Definição do problema CPP
# ================================================================
class MyProblem(ElementwiseProblem):

    def __init__(self):
        super().__init__(
            n_var=3,              # número de variáveis de decisão
            n_obj=2,              # número de objetivos
            n_constr=0,           # número de restrições
            xl=np.array([4.5, 0, 6]),  # limites inferiores
            xu=np.array([ 6.5, 360, 8])   # limites superiores
        )

    def _evaluate(self, x, out, *args, **kwargs):
        # Funções objetivo
        #f1 = x[0]**2 + x[1]**2
        #f2 = (x[0]-1)**2 + (x[1]-1)**2
        f1,P,f2=CPP(x)
        f1=f1
        out["F"] = [f1, f2]
        
        
def rotated_bounding_box(vertices,theta):
    
   
    theta_rad = np.deg2rad(theta)  # converte para radianos
    # --- Matriz de rotação ---
    rotation_matrix = np.array([[np.cos(theta_rad), -np.sin(theta_rad)],[np.sin(theta_rad),  np.cos(theta_rad)]])
    # --- Rotacionar os vértices ---
    rotated_vertices = (rotation_matrix @ vertices.T).T
       
    # --- Plotar o polígono rotacionado (opcional) ---
    #fig, ax = plt.subplots()
    #polygon_patch = Polygon(rotated_vertices, closed=True, facecolor='magenta', alpha=0.3, edgecolor='black')
    #ax.add_patch(polygon_patch)
    
    # --- Calcular a bounding box (caixa envolvente) do polígono rotacionado ---
    xmin = np.min(rotated_vertices[:, 0])
    xmax = np.max(rotated_vertices[:, 0])
    ymin = np.min(rotated_vertices[:, 1])
    ymax = np.max(rotated_vertices[:, 1])
    
    # --- Definir os vértices da bounding box no frame rotacionado ---
    bounding_box_rotated = np.array([
        [xmin, ymin],
        [xmax, ymin],
        [xmax, ymax],
        [xmin, ymax]
    ])
    
    # --- Rotacionar a bounding box de volta para a orientação original ---
    rotation_matrix_inv = rotation_matrix.T  # transposta da matriz de rotação
    bounding_box_original = (rotation_matrix_inv @ bounding_box_rotated.T).T
       
    bounding_box_closed = np.vstack([bounding_box_original, bounding_box_original[0]])
    # Plota com linha tracejada azul
    #plt.plot(bounding_box_closed[:, 0], bounding_box_closed[:, 1], 'b--')  # 'b--' é azul tracejado
    #print(f"{bounding_box_original}")
        
    return bounding_box_original    
    
def determinar_xmin_x_max(theta,xi_r,yi_r):
    
    if (0 <= theta < 90) or (270 < theta < 360):
        x_coords_r = np.sort(xi_r)
        
        if not np.all(np.diff(xi_r) >= 0):  # equivalente a issorted == False
            y_coords_r = np.roll(yi_r, 1)   # circshift(yi_r, 1)
        
        else:
            y_coords_r = yi_r
    if theta == 90:
        y_coords_r = np.sort(yi_r)[::-1]  # ordena em ordem decrescente
        x_coords_r = xi_r

    if theta == 270:
        y_coords_r = np.sort(yi_r)        # ordem crescente padrão
        x_coords_r = xi_r  
    if 90 < theta < 270:
        x_coords_r = np.sort(xi_r)[::-1]  # ordem decrescente
        
        # Verifica se xi_r NÃO está ordenado de forma decrescente
        if not np.all(np.diff(xi_r) <= 0):
            y_coords_r = np.roll(yi_r, 1)  # equivalente ao circshift
            
        else:
            y_coords_r = yi_r 
    return x_coords_r,y_coords_r

def euclidean_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2)) 
def angle_between_lines(a1, a2, b1, b2):
    # Vetores das retas
    v1 = np.array(a2) - np.array(a1)
    v2 = np.array(b2) - np.array(b1)
    
    # Ângulo entre vetores
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))  # em radianos
    return angle

def cost_function(Total_path,id,vel,omega):
    i_end = Total_path.shape[0]
    if id == 1:  # home to point
        home_pos = Total_path[0, :]             # linha 1
        p2 = Total_path[-2, :]                  # penúltima linha
        p3 = Total_path[-1, :]                  # última linha
        
        delta_cost_index = euclidean_distance(home_pos, p2) / vel
        delta_cost_index += euclidean_distance(p2, p3) / vel
        delta_cost_index += angle_between_lines(home_pos, p2, p2, p3) / omega
        
    elif id == 2:  # normal
        p1 = Total_path[-4, :]
        p2 = Total_path[-3, :]
        p3 = Total_path[-2, :]
        p4 = Total_path[-1, :]
        
        delta_cost_index = euclidean_distance(p2, p3) / vel
        delta_cost_index += euclidean_distance(p3, p4) / vel
        delta_cost_index += angle_between_lines(p1, p2, p2, p3) / omega
        delta_cost_index += angle_between_lines(p2, p3, p3, p4) / omega
        
    elif id == 3:  # to home
        p1 = Total_path[-3, :]
        p2 = Total_path[-2, :]
        p3 = Total_path[-1, :]
        delta_cost_index = euclidean_distance(p2, p3) / vel
        delta_cost_index += angle_between_lines(p1, p2, p2, p3) / omega
    
    return delta_cost_index 
    
def CPP(xd): 
    global vertices, home_pos,omega
    
    row_spacing=xd[0]
    theta=xd[1]
    vel=xd[2]
    
    Total_path=0
    cost_index=0
    delta_cost_index=0
    cost_index_parcial=0 #Tempo Trip (Para conferir se a autonomia: cost_index < max_Time_Trip)
    Times_to_home=1 #vezes que o Drone volta para Abstecimento (Carga de bateria)
           
    lim_r = rotated_bounding_box(vertices,theta)
    p1_r = lim_r[0, :]  # 1ª linha → ponto 1 (x, y)
    p2_r = lim_r[1, :]
    p4_r = lim_r[2, :]
    p3_r = lim_r[3, :]
    
    Ac=0
    
    theta_rad = np.deg2rad(theta)  # converte para radianos
    rotation_matrix = np.array([[np.cos(theta_rad), -np.sin(theta_rad)],[np.sin(theta_rad), np.cos(theta_rad)]])
    direction = 1  # 1 for left-to-right, -1 for right-to-left
    
    delta_row=rotation_matrix.T@np.array([[0],[row_spacing]])
    
    current_r1=p1_r+delta_row.T/2;
    current_r2=p2_r+delta_row.T/2;
    
    ly=np.linalg.norm(p3_r - p1_r) # Cumprimento Limites Rotationados
    i_row=1 # Contador de Linhas
       
    polygon = Polygon(vertices)  # vertices deve ser (N, 2) array-like # Construa o polígono
    while np.linalg.norm(current_r1 - p1_r)<=ly:
        #Linha que atraveza o Poligono
        line_x_r = np.array([current_r1[0,0], current_r2[0,0]])
        line_y_r = np.array([current_r1[0,1], current_r2[0,1]])
        # Construa a linha
        line = LineString(zip(line_x_r, line_y_r))  # line_x_r e line_y_r são listas ou arrays com 2 pontos
        # Interseção Poligono e linhas-----------------------------------------
        intersection = line.intersection(polygon)
        xi_r, yi_r = zip(*intersection.coords)
        xi_r = np.array(xi_r)
        yi_r = np.array(yi_r)
        #print(f"intersection: {xi_r} e {yi_r}")
        # Extrair coordenadas da(s) interseção(ões)
        #----------------------------------------------------------------------
        if xi_r.size>0:  # verdadeiro se xi_r não estiver vazio
            x_coords_r,y_coords_r=determinar_xmin_x_max(theta,xi_r,yi_r)
            if direction==1:
                row_points_r = np.array([[x_coords_r[0],y_coords_r[0]],[x_coords_r[-1], y_coords_r[-1]]])
            else:
                row_points_r = np.array([[x_coords_r[-1],y_coords_r[-1]],[x_coords_r[0], y_coords_r[0]]])
            direction = -direction # Inverter direction da direita -> esquerda
            
            if i_row==1:
                
                home_pos = np.array(home_pos).reshape(1, 2)  # se ainda for 1D
                Total_path = np.vstack([home_pos, row_points_r])
                #--------------------------------------------------------------
                #Cost Fuction
                id_c=1  #Add Home -> two point
                delta_cost_index=cost_function(Total_path,id_c,vel,omega);
                cost_index+=delta_cost_index
                cost_index_parcial=cost_index
                current_r1=current_r1+delta_row.T
                current_r2=current_r2+delta_row.T
                #--------------------------------------------------------------
            else:
                ######Total_path_t = np.vstack([Total_path, row_points_r])
                #Cost Function-------------------------------------------------
                Total_path = np.vstack([Total_path, row_points_r])
                id_c=2  #Add two points
                delta_cost_index=cost_function(Total_path,id_c,vel,omega)       
                cost_index_parcial+=delta_cost_index
                
                current_r1=current_r1+delta_row.T
                current_r2=current_r2+delta_row.T
                
                
                
                ######if cost_index_parcial<= max_Time_Trip: # Conferir a autonomia máxima
                ######    cost_index+=delta_cost_index
                ######    Total_path = np.vstack([Total_path, row_points_r])
                ######    current_r1=current_r1+delta_row.T
                ######    current_r2=current_r2+delta_row.T
                ######    #--------------------------------------------------------------
                ######else:             
                ######    ##
                ######    cost_index_parcial-=delta_cost_index
                ######    
                ######    Total_path = np.vstack([Total_path,home_pos])
                ######    Times_to_home+=1
                ######    #Cost Function---------------------------------------------
                ######    id_c=3 #Add to Home
                ######    delta_cost_index=cost_function(Total_path,id_c,vel,omega)
                ######    cost_index+=delta_cost_index
                ######    print(f"Cost function partcial: {cost_index_parcial}")
                ######    cost_index_parcial+=delta_cost_index
                ######    cost_index_parcial=0
                ######    #----------------------------------------------------------
            #print(f"Linha {i_row}: {row_points_r}")
            p1=row_points_r[0,:];
            p2=row_points_r[1,:];
            L1=euclidean_distance(p1, p2)
            Ac+=L1*row_spacing        
            i_row+=1        
    P=.5*1.2*vel**3*0.2      #Potencia
    P_hover=1243 
    E_hover=P_hover*cost_index_parcial/3600
    #print(E_hover)
    E=P*cost_index_parcial/3600#+P_hover*cost_index_parcial/3600   #Energia total 
    #print(Total_path[-1,:])  
    #print(home_pos)  
    #print(f"Cost function partcial: {cost_index_parcial}") # ultima sotrie
    if not np.allclose(Total_path[-1,:],home_pos):
        Total_path = np.vstack([Total_path, home_pos]) #Drone volta para home 
          
    #print(f"Area Coverta: {Ac}")            
    #print(Total_path)
    return cost_index_parcial,Total_path,E

    
def line_sweep_direction(polygon):
    if np.all(polygon[0] == polygon[-1]):
        polygon = polygon[:-1]

    n = polygon.shape[0]
    optimal_dist = np.inf
    sweep_dir = np.array([0.0, 0.0])

    for i in range(n):
        v1 = polygon[i]
        v2 = polygon[(i + 1) % n]
        edge_vec = v2 - v1

        edge_normal = np.array([-edge_vec[1], edge_vec[0]])
        edge_normal = edge_normal / np.linalg.norm(edge_normal)

        max_dist_edge = 0.0
        for j in range(n):
            v = polygon[j]
            dist = abs(np.dot((v - v1), edge_normal))
            if dist > max_dist_edge:
                max_dist_edge = dist

        if max_dist_edge < optimal_dist:
            optimal_dist = max_dist_edge
            sweep_dir = edge_normal

    # ângulo entre vetor de varredura e o eixo Y
    ang = np.degrees(np.arctan2(sweep_dir[0], sweep_dir[1]))  # eixo Y
    ang = (ang + 180) % 180  # limitar entre 0 e 180°

    return ang
def compute_spacing(F):
    # F is the array of objective values (res.F)
    if len(F) <= 1:
        return 0
    
    # Calculate distances to all other points for each point
    # Then find the minimum distance to any other point
    d = []
    for i in range(len(F)):
        # Euclidean distance to all other points
        dist = np.sqrt(np.sum((F - F[i])**2, axis=1))
        # Mask the distance to itself (0)
        dist = np.delete(dist, i)
        d.append(np.min(dist))
    
    d_mean = np.mean(d)
    # The Spacing metric formula
    spacing = np.sqrt(np.sum((d_mean - d)**2) / (len(F) - 1))
    return spacing
    

# ================================================================
# Execução principal
# ================================================================
def main():

    global vertices,home_pos,omega
    
    xd=np.zeros(3)
    
    #-------------------------------------------------------------
    #Torres
    ang_torres =line_sweep_direction(vertices)
    xd[0]=5.5           #delta_r
    xd[1]=ang_torres    #theta
    xd[2]=7             #v
    print(f"Angulo Torres: {ang_torres}")
    f1_torres,P,f2_torres=CPP(xd)
    #-------------------------------------------------------------
    #-------------------------------------------------------------
    #choset
    xd[0]=5.5    #delta_r
    xd[1]=90     #theta
    xd[2]=7      #v
    f1_choset,P,f2_choset=CPP(xd)
    #-------------------------------------------------------------
    
    # Criar problema
    problem = MyProblem()

    #-------------------------------------------------------------
    
    algorithm_NSGA2 = NSGA2(
        pop_size=100,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True
    )
    #--------------------------------------------------------------
    
    ref_dirs = get_reference_directions("uniform", 2, n_partitions=99)

    algorithm_MOEAD = MOEAD(
    ref_dirs=ref_dirs,
    n_neighbors=15,
    prob_neighbor_mating=0.7,
    sampling=FloatRandomSampling(),
    crossover=SBX(prob=0.9, eta=15),
    mutation=PM(eta=20)
    )
    #--------------------------------------------------------------
    
    algorithm_SPEA2 = SPEA2(
    pop_size=100,
    sampling=FloatRandomSampling(),
    crossover=SBX(prob=0.9, eta=15),
    mutation=PM(eta=20),
    eliminate_duplicates=True
    )
    
    #--------------------------------------------------------------
    
    start = time.perf_counter()   
    res = minimize(problem,
                   algorithm_NSGA2,
                   ('n_gen', 100),
                   seed=1,
                   save_history=True, # Critical for metrics
                   verbose=True)
    end = time.perf_counter()
    time_NSGA2=end - start
                
    
    start = time.perf_counter()   
    res_MOEAD = minimize(problem,
                   algorithm_MOEAD,
                   ('n_gen', 100),
                   seed=1,
                   save_history=True, # Critical for metrics
                   verbose=True)
    end = time.perf_counter()
    time_MOEAD=end - start               
    
    
    start = time.perf_counter()   
    res_SPEA2 = minimize(problem,
                   algorithm_SPEA2,
                   ('n_gen', 100),
                   seed=1,
                   save_history=True, # Critical for metrics
                   verbose=True)
                   
    end = time.perf_counter()
    time_SPEA2=end - start 
                   
    indices = np.argsort(res.F[:,0])
    res_F_ordenado = res.F[indices[::10],:]
    
       
    print(f"resposta:{res.X[indices[::10],:] }")
    print(f"Objetivos:{res.F[indices[::10],:]}")
    
    a=res.F.shape
    
    print(f"Size:{a }")
    
    plt.figure(figsize=(6.5, 5.5))
    plt.plot(res.F[:,0],res.F[:,1],'bo',markersize=6, label="NSGA2")
    
    plt.plot(res_MOEAD.F[:,0],res_MOEAD.F[:,1],'m*',markersize=6, label="MOEAD")
    
    plt.plot(res_SPEA2.F[:,0],res_SPEA2.F[:,1],'ys',markersize=6, label="SPEA2")
        
    # Adicionar labels F_i para os 10 primeiros pontos amostrados
    for i, (x, y) in enumerate(res.F[indices[::10]]):  # pega até 10 pontos
        plt.annotate(
        f"$F_{i+1}$",           # texto em LaTeX
        (x, y),                 # ponto (x,y)
        textcoords="offset points",
        xytext=(5, 5),          # deslocamento do texto
        fontsize=14,              # <<< muda o tamanho do texto
        arrowprops=dict(arrowstyle="->", lw=0.8)  # seta
        )
    plt.plot(f1_torres,f2_torres,'rD',markersize=6, label="Torres")
    plt.plot(f1_choset,f2_choset,'g^',markersize=6, label="Choset")
     
    plt.xlabel(r"$c(\lambda)\; [s]$", fontsize=17)
    plt.ylabel(r"$E_{Drag}(\lambda)\; [wh]$", fontsize=17)
    plt.tick_params(axis='both', labelsize=14)  # Tamanho dos números nos eixos X e Y
    
    #plt.tick_params(axis='both', labelsize=13)  # Tamanho dos números nos eixos X e Y
    plt.legend(loc="lower left", fontsize=15)  # Você pode mudar a posição: 'upper left', 'lower 
    #plt.axis('equal')  # Opcional, para manter proporções reais
    plt.show()
    
    #Scatter().add(res.F).show()
    
    #=============================================================   
    print("Computational Effort Evaluation--------------------------------")
    print("NSGA2")
    # Define a reference point (slightly worse than the expected maximums)
    # For your paper: [max mission time, max energy]
    ref_point = np.array([500.0, 6.0]) 
    
    # Extract the Pareto front from each generation
    hist = res.history
    n_gen = len(hist)
    
    hv_list = []
    indicator = HV(ref_point=ref_point)
    
    for generation in hist:
        opt = generation.opt
        if len(opt) > 0:
            hv_list.append(indicator.do(opt.get("F")))
    print(f"Hypervolume generation 100: {hv_list[-1]}")
    
    
    # If you assume the best results from all runs are the 'true' front
    true_front = res.F 
    metric = IGD(true_front)
    
    igd_values = [metric.do(gen.opt.get("F")) for gen in res.history]
    print(f"Inverted Generational Distance (IGD) / generation 100: {igd_values[-1]}")
    
    print(f"Elapsed time: {time_NSGA2:.6f} seconds")   
    
    spc=compute_spacing(res.F)
    
    print(f"Spacing NASGA2: {spc}")
    
    print("MOEAD")
    # Define a reference point (slightly worse than the expected maximums)
    # For your paper: [max mission time, max energy]
    ref_point = np.array([500.0, 6.0]) 
    
    # Extract the Pareto front from each generation
    hist = res_MOEAD.history
    n_gen = len(hist)
    
    hv_list = []
    indicator = HV(ref_point=ref_point)
    
    for generation in hist:
        opt = generation.opt
        if len(opt) > 0:
            hv_list.append(indicator.do(opt.get("F")))
    print(f"Hypervolume generation 100: {hv_list[-1]}")
    
    
    # If you assume the best results from all runs are the 'true' front
    true_front = res_MOEAD.F 
    metric = IGD(true_front)
    
    igd_values = [metric.do(gen.opt.get("F")) for gen in res_MOEAD.history]
    print(f"Inverted Generational Distance (IGD) / generation 100: {igd_values[-1]}")
    
    print(f"Elapsed time: {time_MOEAD*1000:.6f} miliseconds")   
    
    
    spc=compute_spacing(res_MOEAD.F)
    
    print(f"Spacing _MOEAD: {spc}")
    
    
    print("SPEA2")
    # Define a reference point (slightly worse than the expected maximums)
    # For your paper: [max mission time, max energy]
    ref_point = np.array([500.0, 6.0]) 
    
    # Extract the Pareto front from each generation
    hist = res_SPEA2.history
    n_gen = len(hist)
    
    hv_list = []
    indicator = HV(ref_point=ref_point)
    
    for generation in hist:
        opt = generation.opt
        if len(opt) > 0:
            hv_list.append(indicator.do(opt.get("F")))
    print(f"Hypervolume generation 100: {hv_list[-1]}")
    
    
    # If you assume the best results from all runs are the 'true' front
    true_front = res_SPEA2.F 
    metric = IGD(true_front)
    
    igd_values = [metric.do(gen.opt.get("F")) for gen in res.history]
    print(f"Inverted Generational Distance (IGD) / generation 100: {igd_values[-1]}")
    
    print(f"Elapsed time: {time_SPEA2*1000:.6f} miliseconds")   
    
    spc=compute_spacing(res_SPEA2.F)
    
    print(f"Spacing _SPEA2: {spc}")
    
    
    
    
    
    print("-----------------------------------------------------------")
    #=============================================================
     
    xd_i=50   # X6
    row_spacing = res.X[indices[xd_i],0] #m  % X6
    theta=res.X[indices[xd_i],1]
    vel=res.X[indices[xd_i],2] #Velocidade Linear do Drone (m/s) [10 km/h]
    
    
    print("1. MOO CPP Solition:")
    print("2. Torres CPP Solution:")
    print("3. Solution CPP - Case Study - Satallete:")
    
    
    try:
        choice_sol = int(input("Enter your choice (1-2): "))
    except ValueError:
        choice = 0 
    if choice_sol==2: #Torres
        #-------------------------------------------------------------------
        #-------------------------------------------------------------------
        #Torres
        xd[0]=5.5           #delta_r
        xd[1]=ang_torres    #theta
        xd[2]=7             #v
        #-------------------------------------------------------------------
    if choice_sol==1: #MOO
        #ini_cost_index=0;
        xd=np.zeros(3)
            
        xd[0]=row_spacing
        xd[1]=theta
        xd[2]=vel
    if choice_sol==3:
        #ini_cost_index=0;
        xd=np.zeros(3)
            
        xd[0]=row_spacing
        xd[1]=90#theta
        xd[2]=vel 
                   
    
    cost_index,Total_path,Ac=CPP(xd)
    if choice_sol==1 or choice_sol==2:
        # Plota com linha tracejada azul
        plt.figure(figsize=(6.5, 5.5))
        plt.plot(Total_path[:, 0], Total_path[:, 1], 'b-')  # 'b--' é azul tracejado
        plt.plot(vertices[:, 0], vertices[:, 1], 'k-')  # 'b--' é azul tracejado
        #plt.grid(True)     # Opcional, para visualização melhor
        plt.xlabel("x [m]", fontsize=18)          # Rótulo do eixo X
        plt.ylabel("y [m]", fontsize=18)          # Rótulo do eixo Y
        plt.plot(home_pos[0,0], home_pos[0,1], marker='^', color='orange', markersize=14, label="Home Position")
        plt.plot(Total_path[1, 0], Total_path[1, 1], marker='o', color='orange', markersize=14, label="Start")
        plt.plot(Total_path[-2, 0], Total_path[-2, 1], marker='s', color='orange', markersize=14, label="Finish")
        plt.legend(loc="upper right", fontsize=16)  # Legenda com fonte ajustada
        plt.tick_params(axis='both', labelsize=15)  # Tamanho dos números nos eixos X e Y
        
        plt.legend(loc="upper right",fontsize=16)  # Você pode mudar a posição: 'upper left', 'lower right', etc.
        plt.axis('equal')  # Opcional, para manter proporções reais
        #caminho_local = os.path.expanduser("~/Total_path.png")
           
        xmin=-200 
        xmax=300
        #########plt.xlim([xmin, xmax])  # Limita o eixo X
        #plt.savefig(caminho_local, dpi=300)
    if choice_sol==3:
        
           
        # Caminho até o arquivo .shp
        caminho_shp = '~/MOO_CPP_python/area_UFTM.shp'
        # Lê o shapefile como GeoDataFrame
        gdf = gpd.read_file(caminho_shp)
        # Reprojetar para EPSG:3857 (necessário para o mapa base do contextily)
        gdf_proj = gdf.to_crs(epsg=3857)
        # Plotar area
        ax = gdf_proj.plot(figsize=(7.5, 5), alpha=0.0, edgecolor='red')
        ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery,zoom=18) #plotar Mapa # valores típicos: 15 (cidade), 17–19 (bairro/terreno)
        plt.plot(Total_path[:, 0], Total_path[:, 1], 'b-')  # 'b--' é azul tracejado	
        #plt.plot(vertices[:, 0], vertices[:, 1], 'k-')  # 'b--' é azul tracejado
        plt.plot(home_pos[0,0], home_pos[0,1], marker='^', color='orange', markersize=10, label="Home Position")
        plt.plot(Total_path[1, 0], Total_path[1, 1], marker='o', color='orange', markersize=10, label="Start")
        plt.plot(Total_path[-2, 0], Total_path[-2, 1], marker='s', color='orange', markersize=10, label="Finish")
        x_pos = gdf_proj.total_bounds[0] + 0.05 * (gdf_proj.total_bounds[2] - gdf_proj.total_bounds[0])
        y_pos = gdf_proj.total_bounds[1] + 0.95 * (gdf_proj.total_bounds[3] - gdf_proj.total_bounds[1])
        ax.annotate('N', xy=(x_pos, y_pos), xytext=(x_pos, y_pos + 0.05),
            arrowprops=dict(facecolor='red', width=4, headwidth=20),
            ha='center', fontsize=22, fontweight='bold',color='red')
        plt.legend(loc="upper right", fontsize=14)  # Legenda com fonte ajustada
        plt.tick_params(axis='both', labelsize=14)  # Tamanho dos números nos eixos X e Y     
    
    plt.show()       
    


    
# ================================================================
# Ponto de entrada
# ================================================================
if __name__ == "__main__":
    main()
