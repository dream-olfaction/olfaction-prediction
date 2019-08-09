Assuming you have Docker:

1. Clone this repository and `cd` to this directory (`docker`).  
   - Or just put the Dockerfile in an empty directory and `cd` there.
2. `docker build -t olfaction-prediction . --no-cache`
3. `docker run -it -p 8889:8888 olfaction-prediction`
4. Go to `localhost:8889` in your browser.
5. Execute all the cells in the notebook.
