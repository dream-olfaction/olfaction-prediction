FROM jupyter/datascience-notebook

#RUN git clone -b thin https://github.com/dream-olfaction/olfaction-prediction
RUN wget http://github.com/dream-olfaction/olfaction-prediction/tarball/thin -O olfaction-prediction.tar.gz
RUN mkdir olfaction-prediction 
RUN tar -xvzf olfaction-prediction.tar.gz -C olfaction-prediction --strip-components=1
WORKDIR olfaction-prediction
RUN pip install -e .
RUN jupyter nbconvert --to python opc_python/paper/quick-start.ipynb
ENTRYPOINT jupyter notebook --NotebookApp.token=''