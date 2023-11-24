{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "include_colab_link": True
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/Juanafenley/avanzadoIA/blob/main/pruebaStreamlit.ipynb\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "!pip install -q streamlit"
      ],
      "metadata": {
        "id": "3Be3vO1rl-n8",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "2509d6c1-f447-424e-ac79-e9aeb6a0998d"
      },
      "execution_count": None,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m8.4/8.4 MB\u001b[0m \u001b[31m42.5 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m190.6/190.6 kB\u001b[0m \u001b[31m19.4 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m4.8/4.8 MB\u001b[0m \u001b[31m81.5 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m82.1/82.1 kB\u001b[0m \u001b[31m9.3 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m62.7/62.7 kB\u001b[0m \u001b[31m7.1 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25h"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {
        "id": "jSkUr8k1kDoc"
      },
      "outputs": [],
      "source": [
        "from tensorflow.keras.models import load_model\n",
        "from tensorflow.keras.applications.imagenet_utils import preprocess_input\n",
        "import numpy as np\n",
        "import streamlit as st\n",
        "from PIL import Image\n",
        "from google.colab import drive\n",
        "from skimage.transform import resize"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "drive.mount('/content/drive')"
      ],
      "metadata": {
        "id": "bYDHOvS-4txX",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "f7825101-5daa-44ab-98b3-9ad8b486b9b5"
      },
      "execution_count": None,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Mounted at /content/drive\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "names=[\"Elefante\", \"Mariposa\",\"Vaca\", \"Oveja\",\"Ardilla\"]"
      ],
      "metadata": {
        "id": "QVRhuE4Y3sUM"
      },
      "execution_count": None,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "def model_prediccion(img, model):\n",
        "  img_resize=resize(img, (224,224))\n",
        "  x= preprocess_input(img_resize*225)\n",
        "  x=np.expand_dims(x, axis=0)\n",
        "\n",
        "  preds=model.predict(x)\n",
        "  return preds\n"
      ],
      "metadata": {
        "id": "B7tEEFCdmLrp"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "def main():\n",
        "  modelo=load_model(\"/content/drive/MyDrive/DATASET/modelo_animales1.h5/\")\n",
        "  st.title(\"Clasificador de Animales\")\n",
        "  img_file_buffer=st.file_uploader(\"Cargar una imagen\", type=[\"png\", \"jpg\", \"jpeg\"])\n",
        "  if img_file_buffer is not None:\n",
        "    image=np.array(Image.open(img_file_buffer))\n",
        "    st.image(image, caption=\"Imagen\", use_column_width=False)\n",
        "  if st.button(\"Predicción\"):\n",
        "    predict=model_prediccion(image, modelo)\n",
        "    st.success(\"La clase es:\".format(names[np.argmax(predict)]))"
      ],
      "metadata": {
        "id": "vdK7K4f8oEi3"
      },
      "execution_count": None,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "main()"
      ],
      "metadata": {
        "id": "rDlYmWyd4VkZ",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "e9e4b0ad-df72-428f-c0ab-810c4e75d9a0"
      },
      "execution_count": None,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "2023-11-24 01:10:41.245 \n",
            "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
            "  command:\n",
            "\n",
            "    streamlit run /usr/local/lib/python3.10/dist-packages/colab_kernel_launcher.py [ARGUMENTS]\n"
          ]
        }
      ]
    }
  ]
}
