DOCKERNAME=thornado-ad
# build docker if not exists
if [ -z "$(docker images -q $DOCKERNAME)" ] || [ "$FORCE_BUILD" = true ]; then
    echo "Building docker image"
    docker build -t $DOCKERNAME .
fi

echo "Running docker image"
# COMMAND="./test.sh"
# COMMAND="/bin/bash"
# docker run -v "$PWD/mount:/root/ad" -it --rm -t $DOCKERNAME /bin/bash -c "cd /root/ad && $COMMAND"

# docker run -v "$PWD/result:/home/s8maullr/results" -it --rm -t $DOCKERNAME 
docker run -v "$PWD/tmp:/home/" -it --rm -t $DOCKERNAME 

