
## Building standalone motor test
docker build -t testdcmotor .

docker run --rm -p 8000:8000 -v /tmp:/app/saved -e NUM_PLANTS=2 -e SAMPLE_PERIOD=0.02 testdcmotor

