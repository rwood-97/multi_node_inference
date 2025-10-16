# Hosting a frontend

This directory contains scripts to run [Open-WebUI](https://docs.vllm.ai/en/latest/deployment/frameworks/open-webui.html) as a frontend for your model.

To do this, you will need to forward ports from the HPC compute node (or wherever you ran the `vllm serve` command) to your local machine.

## Baskerville

On Baskerville, you can use the following command to forward port 8000 on the compute node to port 8000 on your local machine:

```bash
ssh -t -t uuname@login.baskerville.ac.uk -L 8000:localhost:8000 ssh bask-pgXXXXXXX -L 8000:localhost:8000
```

You will need to replace `uuname` with your username and `bask-pgXXXXXXX` with the actual compute node name. You can find this by running `squeue --me` and looking at the `NODELIST` column.

## Isambard-AI

On Isambard-AI, you can use the following command to forward port 8000 on the compute node to port 8000 on your local machine:

```bash
ssh -T -L localhost:8000:x.x.x.x:8000 proj.aip2.isambard
```

You will need to replace `proj` with your project id and `x.x.x.x` with the IP address of the head node (this is printed as `Primary IP: x.x.x.x` in the logs).

## Running Open-WebUI

Once you've set up the port forwarding, you can check it works by going to `http://localhost:8000/v1/models` in your browser.
Then, you can simply run the `run_frontend.sh` script to start Open-WebUI.

### Troubleshooting

If you have issues getting your model to show up in the model list, you may need to:

1. Go to the "Admin Panel"
2. Go to "Settings" -> "Connections"
3. Check your API url is set in the "OpenAI API" section, you can also check it is working by clicking the "Verify Connection" button
4. (Maybe?) Ensure that the "Direct Connections" toggle is enabled
