cmake_minimum_required(VERSION 3.0.2)
project(sim_pkg)

## Encuentra los paquetes catkin necesarios
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
)

## Declara el paquete catkin
catkin_package(
  CATKIN_DEPENDS roscpp rospy std_msgs
)

## Incluye directorios
include_directories(
  ${catkin_INCLUDE_DIRS}
)

## Opcional: Si tienes nodos en C++ agrega un ejecutable
# add_executable(${PROJECT_NAME}_node src/main.cpp)
# target_link_libraries(${PROJECT_NAME}_node ${catkin_LIBRARIES})

## Instala los archivos de lanzamiento
install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)
