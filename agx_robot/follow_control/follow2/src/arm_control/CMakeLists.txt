cmake_minimum_required(VERSION 3.0.2)
project(arm_control)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CURSES_NEED_NCURSES TRUE)
find_package(catkin REQUIRED COMPONENTS
  dynamic_reconfigure
  geometry_msgs
  roscpp
  rospy
  std_msgs
  tf
  message_generation
  message_runtime
  kdl_parser
)


#####################ecat############################
if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  # Default to installing in SOEM source directory
  set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_LIST_DIR}/install)
endif()

set(SOEM_INCLUDE_INSTALL_DIR ${CATKIN_DEVEL_PREFIX}/include/${PROJECT_NAME})
set(SOEM_LIB_INSTALL_DIR ${CATKIN_DEVEL_PREFIX}/lib/${PROJECT_NAME})



if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  set(BUILD_TESTS TRUE)
else()
  message(STATUS "SOEM: not building tests when built as dependency")
  set(BUILD_TESTS FALSE)
endif()


if(WIN32)
  set(OS "win32")
  include_directories(oshw/win32/wpcap/Include)
  if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    link_directories(${CMAKE_CURRENT_LIST_DIR}/oshw/win32/wpcap/Lib/x64)
  elseif(CMAKE_SIZEOF_VOID_P EQUAL 4)
    link_directories(${CMAKE_CURRENT_LIST_DIR}/oshw/win32/wpcap/Lib)
  endif()
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /D _CRT_SECURE_NO_WARNINGS")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  /WX")
  set(OS_LIBS wpcap.lib Packet.lib Ws2_32.lib Winmm.lib)
elseif(UNIX AND NOT APPLE)
  set(OS "linux")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -Werror")
  set(OS_LIBS pthread rt)
elseif(APPLE)
  # This must come *before* linux or MacOSX will identify as Unix.
  set(OS "macosx")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -Werror")
  set(OS_LIBS pthread pcap)
elseif(${CMAKE_SYSTEM_NAME} MATCHES "rt-kernel")
  set(OS "rtk")
  message(STATUS "ARCH is ${ARCH}")
  message(STATUS "BSP is ${BSP}")
  include_directories(oshw/${OS}/${ARCH})
  file(GLOB OSHW_EXTRA_SOURCES oshw/${OS}/${ARCH}/*.c)
  set(OSHW_SOURCES "${OS_HW_SOURCES} ${OSHW_ARCHSOURCES}")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -Werror")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-unused-but-set-variable")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-unused-function")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-format")
  set(OS_LIBS "-Wl,--start-group -l${BSP} -l${ARCH} -lkern -ldev -lsio -lblock -lfs -lusb -llwip -leth -li2c -lrtc -lcan -lnand -lspi -lnor -lpwm -ladc -ltrace -lc -lm -Wl,--end-group")
elseif(${CMAKE_SYSTEM_NAME} MATCHES "rtems")
  message(STATUS "Building for RTEMS")
  set(OS "rtems")
  set(SOEM_LIB_INSTALL_DIR ${LIB_DIR})
  set(BUILD_TESTS FALSE)
endif()


message(STATUS "OS is ${OS}")

file(GLOB SOEM_SOURCES soem/*.c)
file(GLOB OSAL_SOURCES osal/${OS}/*.c)
file(GLOB OSHW_SOURCES oshw/${OS}/*.c)

file(GLOB SOEM_HEADERS soem/*.h)
file(GLOB OSAL_HEADERS osal/osal.h osal/${OS}/*.h)
file(GLOB OSHW_HEADERS oshw/${OS}/*.h)

file(GLOB SOEM_HEADERS_DIR soem)
file(GLOB OSAL_HEADERS_DIR osal osal/${OS})
file(GLOB OSHW_HEADERS_DIR oshw/${OS})


add_library(soem STATIC
  ${SOEM_SOURCES}
  ${OSAL_SOURCES}
  ${OSHW_SOURCES}
  ${OSHW_EXTRA_SOURCES})
target_link_libraries(soem ${OS_LIBS})


target_include_directories(soem PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/soem>
  $<INSTALL_INTERFACE:include/soem>)

target_include_directories(soem PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/osal>
  $<INSTALL_INTERFACE:include/soem>)

target_include_directories(soem PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/osal/${OS}>
  $<INSTALL_INTERFACE:include/soem>)

target_include_directories(soem
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/oshw/${OS}>
  $<INSTALL_INTERFACE:include/soem>
  )


  message(STATUS "LIB_DIR: ${SOEM_LIB_INSTALL_DIR}")

  install(TARGETS soem EXPORT soemConfig DESTINATION ${SOEM_LIB_INSTALL_DIR})
  
  install(EXPORT soemConfig DESTINATION share/soem/cmake)
  
  install(FILES
    ${SOEM_HEADERS}
    ${OSAL_HEADERS}
    ${OSHW_HEADERS}
    DESTINATION ${SOEM_INCLUDE_INSTALL_DIR})
  

#################################################



add_message_files(
  FILES
  PosCmd.msg
  JointControl.msg 
  JointInformation.msg 
  ChassisCtrl.msg
  MagicCmd.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs#   std_msgs
  # arx
)

generate_dynamic_reconfigure_options(
  cfg/reconfig.cfg
#   cfg/DynReconf1.cfg
#   cfg/DynReconf2.cfg
)

catkin_package(
 INCLUDE_DIRS include
 CATKIN_DEPENDS dynamic_reconfigure geometry_msgs roscpp rospy std_msgs tf #message_generation message_runtime
 CATKIN_DEPENDS kdl_parser
)

include_directories(
include
  ${catkin_INCLUDE_DIRS}
  ${roscpp_INCLUDE_DIRS}

#########Ecat###################
${SOEM_HEADERS_DIR}
${OSAL_HEADERS_DIR}
${OSHW_HEADERS_DIR}
${catkin_INCLUDE_DIRS}
############################


)

add_library(arm_control
   SHARED
   src/App/arm_control.cpp
   src/Hardware/teleop.cpp

)

add_executable(follow_2 src/arx.cpp
# >>>Ecat######################
${SOEM_SOURCES}
${OSAL_SOURCES}
${OSHW_SOURCES}
${OSHW_EXTRA_SOURCES}
# >>>Ecat######################


)
add_dependencies(follow_2 ${PROJECT_NAME}_gencfg)


if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm.*|arm|aarch64")
    # Link libraries for ARM architecture
    target_link_libraries(follow_2 
        ${catkin_LIBRARIES} 
        arm_control 

        kdl_parser
    )
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64)|(AMD64)|(^i686)")
    # Link libraries for x86 architecture
    target_link_libraries(follow_2 
        ${catkin_LIBRARIES} 
        arm_control 
        ${CMAKE_CURRENT_SOURCE_DIR}/src/arx_lib/x86/libplay.so 
        ${CMAKE_CURRENT_SOURCE_DIR}/src/arx_lib/x86/libkey.so

        ${CMAKE_CURRENT_SOURCE_DIR}/src/arx_lib/x86/libarx_can.so
        ${CMAKE_CURRENT_SOURCE_DIR}/src/arx_lib/x86/libarx_s.so

        ${CMAKE_CURRENT_SOURCE_DIR}/src/arx_lib/x86/libarx_ecat.so
        ${CMAKE_CURRENT_SOURCE_DIR}/src/arx_lib/x86/libarx_h2.so
        ${CMAKE_CURRENT_SOURCE_DIR}/src/arx_lib/x86/libarx_h3.so
        ${CMAKE_CURRENT_SOURCE_DIR}/src/arx_lib/x86/libarx_l4.so
        ${CMAKE_CURRENT_SOURCE_DIR}/src/arx_lib/x86/libarx_solve.so

        kdl_parser
    )
endif()

install(TARGETS follow_2
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(TARGETS arm_control
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}
)

install(DIRECTORY
  ${CMAKE_CURRENT_SOURCE_DIR}/launch
  ${CMAKE_CURRENT_SOURCE_DIR}/models
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)