execute_process(COMMAND "/home/ari/catkin_ws/build/new_packageDir/perception/catkin_generated/python_distutils_install.sh" RESULT_VARIABLE res)

if(NOT res EQUAL 0)
  message(FATAL_ERROR "execute_process(/home/ari/catkin_ws/build/new_packageDir/perception/catkin_generated/python_distutils_install.sh) returned error code ")
endif()
