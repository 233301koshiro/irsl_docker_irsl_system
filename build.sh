#!/bin/bash

set -e

# ./build.sh [ target_image_name ]

## arguments as environment variables
# BUILD_ROS [ noetic ] or humble
# REPO [repo.irsl.eiiris.tut.ac.jp/]
# INPUT_IMAGE [ ${REPO}irsl_base:${BUILD_ROS}_nvidia ]
# NO_CACHE [ '' ]

## noetic or melodic
ROS_DISTRO_=${BUILD_ROS:-"noetic"}
CUR_UBUNTU=${UBUNTU_VER:-""}
if [ ${ROS_DISTRO_} == "humble" ]; then
    if [ -z "${CUR_UBUNTU}" ]; then
        CUR_UBUNTU="22.04"
    fi
elif [ ${ROS_DISTRO_} == "jazzy" ]; then
    if [ -z "${CUR_UBUNTU}" ]; then
        CUR_UBUNTU="24.04"
    fi
elif [ ${ROS_DISTRO_} == "one" ]; then
    if [ -z "${CUR_UBUNTU}" ]; then
        #CUR_UBUNTU="22.04"
        CUR_UBUNTU="24.04"
    fi
elif [ ${ROS_DISTRO_} == "noetic" ]; then
    if [ -z "${CUR_UBUNTU}" ]; then
        CUR_UBUNTU="20.04"
    fi
elif [ ${ROS_DISTRO_} == "melodic" ]; then
    if [ -z "${CUR_UBUNTU}" ]; then
        CUR_UBUNTU="18.04"
    fi
fi

DOCKER_OPT='--progress plain'

_REPO=${REPO:-repo.irsl.eiiris.tut.ac.jp/}
XEUS_IMG=${_REPO}xeus:${CUR_UBUNTU}
BASE_IMG=${INPUT_IMAGE:-${_REPO}irsl_base:${ROS_DISTRO_}_opengl}

DEFAULT_IMG=${_REPO}irsl_system:${ROS_DISTRO_}
TARGET_IMG=${1:-${DEFAULT_IMG}}

if [ -n ${NO_CACHE} ]; then
    DOCKER_OPT="--no-cache ${DOCKER_OPT}"
fi

DOCKER_FILE=Dockerfile.build_system.vcstool
if [ -n "${BUILD_DEVEL}" ]; then
    echo "!!!! !!!! Build Devel !!!! !!!!"
    DOCKER_FILE=Dockerfile.build_system.vcstool
fi

echo "Build Image: ${TARGET_IMG}"

set -x

#PULL=--pull
PULL=
docker build . --progress=plain ${PULL} -f Dockerfile.add_xeus  \
       --build-arg BASE_IMAGE=${BASE_IMG} --build-arg BUILD_IMAGE=${XEUS_IMG} \
       -t build_temp/build_system:0

docker build . ${DOCKER_OPT} -f ${DOCKER_FILE} \
       --build-arg BASE_IMAGE=build_temp/build_system:0 \
       -t ${TARGET_IMG}
