Installation Guide (Ascend NPU)
================================

This guide walks through setting up the environment for veRL on Ascend NPU hardware, starting from the official CANN base image. This guide is specifically tested with the CANN 8.3.RC1 base image and is compatible with Qwen3-VL-MoE.

----

Step 0: Pull and Enter the Base Docker Image
--------------------------------------------

Pull the CANN 8.3.RC1 base image and start a container:

.. code-block:: bash

    docker pull swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:8.3.rc1-a3-ubuntu22.04-py3.11
    docker run -it --name verl-env swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:8.3.rc1-a3-ubuntu22.04-py3.11 /bin/bash

.. note::

    All of the following steps should be executed **inside this container**.

----

Step 1: Install System Dependencies
------------------------------------

.. code-block:: bash

    apt-get update -y && \
        apt-get install -y --no-install-recommends \
        gcc g++ cmake libnuma-dev wget git curl jq vim build-essential && \
        apt-get clean && \
        rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

    pip install --upgrade pip packaging setuptools==80.10.2
    pip cache purge

Step 2: Clone Required Repositories
-------------------------------------

.. code-block:: bash

    git clone --depth 1 --branch v0.11.0 https://github.com/vllm-project/vllm.git
    git clone --depth 1 --branch v0.11.0 https://github.com/vllm-project/vllm-ascend.git
    git clone https://gitcode.com/Ascend/MindSpeed.git
    git clone --depth 1 --branch core_v0.14.0 https://github.com/NVIDIA/Megatron-LM.git

Step 3: Set Up Ascend Environment Variables
--------------------------------------------

Depending on your architecture, export the appropriate library path and source the Ascend environment scripts.

**For aarch64:**

.. code-block:: bash

    export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/8.3.RC1/aarch64-linux/devlib/linux/aarch64:$LD_LIBRARY_PATH

Then source the toolkit environment:

.. code-block:: bash

    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    source /usr/local/Ascend/nnal/atb/set_env.sh

Step 4: Install PyTorch and Ascend NPU Support
-----------------------------------------------

.. code-block:: bash

    pip install torch==2.7.1 torch_npu==2.7.1 torchvision==0.22.1 transformers==4.57.6

Step 5: Install vLLM and vLLM-Ascend
--------------------------------------

.. code-block:: bash

    cd vllm && VLLM_TARGET_DEVICE=empty pip install -v -e . && cd ..
    cd vllm-ascend && pip install -v -e . && cd ..

Step 6: Install Megatron-LM and MindSpeed
------------------------------------------

.. code-block:: bash

    cd Megatron-LM && pip install -v -e . && cd ..
    cd MindSpeed && git checkout core_r0.14.0 && pip install -e . && cd ..

Step 7: Clean Up Conflicting Packages
---------------------------------------

Remove any ``triton`` or ``triton-ascend`` that may have been pulled in as transitive dependencies:

.. code-block:: bash

    pip uninstall -y triton triton-ascend

Step 8: Install mbridge
------------------------

Install from the Git repository directly. Commit ``4389fcc450c5f90f0cf22e9c77e3d49e2c643e24`` is confirmed to work:

.. code-block:: bash

    pip install git+https://github.com/ISEEKYAN/mbridge.git@4389fcc450c5f90f0cf22e9c77e3d49e2c643e24
    rm -rf /tmp/* /var/tmp/*
    pip cache purge

Step 9: Install veRL
---------------------

.. code-block:: bash

    git clone https://github.com/HwCARI/verl.git
    cd verl && git checkout ascend-dev
    pip install -r requirements-npu.txt
    pip install -v -e .
    cd ..

Step 10: Install Apex
----------------------

Apex is required for certain training configurations. Refer to the
`official Ascend Apex installation guide <https://www.hiascend.com/document/detail/zh/Pytorch/730/configandinstg/instg/docs/installing_apex.md>`_
for details.

.. code-block:: bash

    git clone -b master https://gitcode.com/Ascend/apex.git
    cd apex
    bash scripts/build.sh --python=3.11
    cd apex/dist/
    pip uninstall -y apex   # optional, only needed if a previous version is installed
    pip install apex-0.1+ascend-cp311-cp311-linux_aarch64.whl

Step 12: Apply Manual Patches to MindSpeed
-------------------------------------------

Two source files in MindSpeed require manual edits.

12a: Register Qwen3-VL Transformer Config Patch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In ``/MindSpeed/mindspeed/features_manager/megatron_basic/megatron_basic.py``,
add the following line at **line 48**:

.. code-block:: python

    pm.register_patch("mbridge.models.qwen3_vl.transformer_config.Qwen3VLTransformerConfig.__init__", transformer_config_init_wrapper)

12b: Fix Bias Assignment in LayerNorm Column Parallel Linear
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In ``/MindSpeed/mindspeed/te/pytorch/module/layernorm_column_parallel_linear.py``,
add the following line at **line 299**:

.. code-block:: python

    bias = self.bias if self.skip_bias_add else None
