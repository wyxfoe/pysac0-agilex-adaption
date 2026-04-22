# Install script for directory: /home/lin/Documents/test/remote_control/follow_control/follow1/src/arm_control

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/lin/Documents/test/remote_control/follow_control/follow1/install")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/lin/Documents/test/remote_control/follow_control/follow1/devel/lib/arm_control/libsoem.a")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
file(INSTALL DESTINATION "/home/lin/Documents/test/remote_control/follow_control/follow1/devel/lib/arm_control" TYPE STATIC_LIBRARY FILES "/home/lin/Documents/test/remote_control/follow_control/follow1/devel/lib/libsoem.a")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/soem/cmake/soemConfig.cmake")
    file(DIFFERENT EXPORT_FILE_CHANGED FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/soem/cmake/soemConfig.cmake"
         "/home/lin/Documents/test/remote_control/follow_control/follow1/build/arm_control/CMakeFiles/Export/share/soem/cmake/soemConfig.cmake")
    if(EXPORT_FILE_CHANGED)
      file(GLOB OLD_CONFIG_FILES "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/soem/cmake/soemConfig-*.cmake")
      if(OLD_CONFIG_FILES)
        message(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/soem/cmake/soemConfig.cmake\" will be replaced.  Removing files [${OLD_CONFIG_FILES}].")
        file(REMOVE ${OLD_CONFIG_FILES})
      endif()
    endif()
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/soem/cmake" TYPE FILE FILES "/home/lin/Documents/test/remote_control/follow_control/follow1/build/arm_control/CMakeFiles/Export/share/soem/cmake/soemConfig.cmake")
  if("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^()$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/soem/cmake" TYPE FILE FILES "/home/lin/Documents/test/remote_control/follow_control/follow1/build/arm_control/CMakeFiles/Export/share/soem/cmake/soemConfig-noconfig.cmake")
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/lin/Documents/test/remote_control/follow_control/follow1/devel/include/arm_control/ethercat.h;/home/lin/Documents/test/remote_control/follow_control/follow1/devel/include/arm_control/ethercatbase.h;/home/lin/Documents/test/remote_control/follow_control/follow1/devel/include/arm_control/ethercatcoe.h;/home/lin/Documents/test/remote_control/follow_control/follow1/devel/include/arm_control/ethercatconfig.h;/home/lin/Documents/test/remote_control/follow_control/follow1/devel/include/arm_control/ethercatconfiglist.h;/home/lin/Documents/test/remote_control/follow_control/follow1/devel/include/arm_control/ethercatdc.h;/home/lin/Documents/test/remote_control/follow_control/follow1/devel/include/arm_control/ethercateoe.h;/home/lin/Documents/test/remote_control/follow_control/follow1/devel/include/arm_control/ethercatfoe.h;/home/lin/Documents/test/remote_control/follow_control/follow1/devel/include/arm_control/ethercatmain.h;/home/lin/Documents/test/remote_control/follow_control/follow1/devel/include/arm_control/ethercatprint.h;/home/lin/Documents/test/remote_control/follow_control/follow1/devel/include/arm_control/ethercatsoe.h;/home/lin/Documents/test/remote_control/follow_control/follow1/devel/include/arm_control/ethercattype.h;/home/lin/Documents/test/remote_control/follow_control/follow1/devel/include/arm_control/osal_defs.h;/home/lin/Documents/test/remote_control/follow_control/follow1/devel/include/arm_control/osal.h;/home/lin/Documents/test/remote_control/follow_control/follow1/devel/include/arm_control/nicdrv.h;/home/lin/Documents/test/remote_control/follow_control/follow1/devel/include/arm_control/oshw.h")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
file(INSTALL DESTINATION "/home/lin/Documents/test/remote_control/follow_control/follow1/devel/include/arm_control" TYPE FILE FILES
    "/home/lin/Documents/test/remote_control/follow_control/follow1/src/arm_control/soem/ethercat.h"
    "/home/lin/Documents/test/remote_control/follow_control/follow1/src/arm_control/soem/ethercatbase.h"
    "/home/lin/Documents/test/remote_control/follow_control/follow1/src/arm_control/soem/ethercatcoe.h"
    "/home/lin/Documents/test/remote_control/follow_control/follow1/src/arm_control/soem/ethercatconfig.h"
    "/home/lin/Documents/test/remote_control/follow_control/follow1/src/arm_control/soem/ethercatconfiglist.h"
    "/home/lin/Documents/test/remote_control/follow_control/follow1/src/arm_control/soem/ethercatdc.h"
    "/home/lin/Documents/test/remote_control/follow_control/follow1/src/arm_control/soem/ethercateoe.h"
    "/home/lin/Documents/test/remote_control/follow_control/follow1/src/arm_control/soem/ethercatfoe.h"
    "/home/lin/Documents/test/remote_control/follow_control/follow1/src/arm_control/soem/ethercatmain.h"
    "/home/lin/Documents/test/remote_control/follow_control/follow1/src/arm_control/soem/ethercatprint.h"
    "/home/lin/Documents/test/remote_control/follow_control/follow1/src/arm_control/soem/ethercatsoe.h"
    "/home/lin/Documents/test/remote_control/follow_control/follow1/src/arm_control/soem/ethercattype.h"
    "/home/lin/Documents/test/remote_control/follow_control/follow1/src/arm_control/osal/linux/osal_defs.h"
    "/home/lin/Documents/test/remote_control/follow_control/follow1/src/arm_control/osal/osal.h"
    "/home/lin/Documents/test/remote_control/follow_control/follow1/src/arm_control/oshw/linux/nicdrv.h"
    "/home/lin/Documents/test/remote_control/follow_control/follow1/src/arm_control/oshw/linux/oshw.h"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/arm_control/msg" TYPE FILE FILES
    "/home/lin/Documents/test/remote_control/follow_control/follow1/src/arm_control/msg/PosCmd.msg"
    "/home/lin/Documents/test/remote_control/follow_control/follow1/src/arm_control/msg/JointControl.msg"
    "/home/lin/Documents/test/remote_control/follow_control/follow1/src/arm_control/msg/JointInformation.msg"
    "/home/lin/Documents/test/remote_control/follow_control/follow1/src/arm_control/msg/ChassisCtrl.msg"
    "/home/lin/Documents/test/remote_control/follow_control/follow1/src/arm_control/msg/MagicCmd.msg"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/arm_control/cmake" TYPE FILE FILES "/home/lin/Documents/test/remote_control/follow_control/follow1/build/arm_control/catkin_generated/installspace/arm_control-msg-paths.cmake")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/home/lin/Documents/test/remote_control/follow_control/follow1/devel/include/arm_control")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/roseus/ros" TYPE DIRECTORY FILES "/home/lin/Documents/test/remote_control/follow_control/follow1/devel/share/roseus/ros/arm_control")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/common-lisp/ros" TYPE DIRECTORY FILES "/home/lin/Documents/test/remote_control/follow_control/follow1/devel/share/common-lisp/ros/arm_control")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/gennodejs/ros" TYPE DIRECTORY FILES "/home/lin/Documents/test/remote_control/follow_control/follow1/devel/share/gennodejs/ros/arm_control")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  execute_process(COMMAND "/home/lin/software/miniconda3/bin/python3" -m compileall "/home/lin/Documents/test/remote_control/follow_control/follow1/devel/lib/python3/dist-packages/arm_control")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/python3/dist-packages" TYPE DIRECTORY FILES "/home/lin/Documents/test/remote_control/follow_control/follow1/devel/lib/python3/dist-packages/arm_control")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/arm_control" TYPE FILE FILES "/home/lin/Documents/test/remote_control/follow_control/follow1/devel/include/arm_control/reconfigConfig.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/python3/dist-packages/arm_control" TYPE FILE FILES "/home/lin/Documents/test/remote_control/follow_control/follow1/devel/lib/python3/dist-packages/arm_control/__init__.py")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  execute_process(COMMAND "/home/lin/software/miniconda3/bin/python3" -m compileall "/home/lin/Documents/test/remote_control/follow_control/follow1/devel/lib/python3/dist-packages/arm_control/cfg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/python3/dist-packages/arm_control" TYPE DIRECTORY FILES "/home/lin/Documents/test/remote_control/follow_control/follow1/devel/lib/python3/dist-packages/arm_control/cfg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/pkgconfig" TYPE FILE FILES "/home/lin/Documents/test/remote_control/follow_control/follow1/build/arm_control/catkin_generated/installspace/arm_control.pc")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/arm_control/cmake" TYPE FILE FILES "/home/lin/Documents/test/remote_control/follow_control/follow1/build/arm_control/catkin_generated/installspace/arm_control-msg-extras.cmake")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/arm_control/cmake" TYPE FILE FILES
    "/home/lin/Documents/test/remote_control/follow_control/follow1/build/arm_control/catkin_generated/installspace/arm_controlConfig.cmake"
    "/home/lin/Documents/test/remote_control/follow_control/follow1/build/arm_control/catkin_generated/installspace/arm_controlConfig-version.cmake"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/arm_control" TYPE FILE FILES "/home/lin/Documents/test/remote_control/follow_control/follow1/src/arm_control/package.xml")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/arm_control/follow_1" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/arm_control/follow_1")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/arm_control/follow_1"
         RPATH "")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/arm_control" TYPE EXECUTABLE FILES "/home/lin/Documents/test/remote_control/follow_control/follow1/devel/lib/arm_control/follow_1")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/arm_control/follow_1" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/arm_control/follow_1")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/arm_control/follow_1"
         OLD_RPATH "/opt/ros/noetic/lib:/home/lin/Documents/test/remote_control/follow_control/follow1/devel/lib:/home/lin/Documents/test/remote_control/follow_control/follow1/src/arm_control/src/arx_lib/x86:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/arm_control/follow_1")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libarm_control.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libarm_control.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libarm_control.so"
         RPATH "")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/home/lin/Documents/test/remote_control/follow_control/follow1/devel/lib/libarm_control.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libarm_control.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libarm_control.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libarm_control.so")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/arm_control" TYPE DIRECTORY FILES
    "/home/lin/Documents/test/remote_control/follow_control/follow1/src/arm_control/launch"
    "/home/lin/Documents/test/remote_control/follow_control/follow1/src/arm_control/models"
    )
endif()

