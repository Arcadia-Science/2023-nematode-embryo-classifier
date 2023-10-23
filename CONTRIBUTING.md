# Contributing Guidelines

Thanks for your intereset in contributing to embryostage.

## Setting up development environment

embryostage is a deep learning pipeline and requires a GPU for efficient training of models. The models we currently use are delibrately light weight to enable efficient iteration. 

We recommend following configuration:
* 128GB RAM
* 16GB GPU RAM
* 32 core x64 CPU
* linux or mac os x environment
* 2TB disk space

Follow [README](README.md) to create and activate the `embryostage` conda environment. 

You can install the dependencies needed for development using:
```sh
    pip install -e .'[dev]'
```

## Setting up remote development environment

Our current remote development environment consists of vscode, vscode remote-ssh extension for remote development, vscode jupyter extension to use the cell mode for interactive computing, vncserver on remote node for GUI, and vnc client. We use napari as a N-dimensional viewer and tensorboard to view the logs of trained models. 

If you are working at Arcadia, see these [instructions](https://docs.google.com/document/d/1FNlo_8fPDrZWld80FSS6C0m-BI61hfqIqdt2aYh_AVY/edit#heading=h.gb4vfu1kxsm2) to use the AWS node dedicated to this project.

### setup ssh and vscode
We strongly recommend setting up [ssh key-based authentication](https://www.digitalocean.com/community/tutorials/how-to-configure-ssh-key-based-authentication-on-a-linux-server) for remote development. 

Once you have copied your *public key* to your account on the remote node, setup your laptop's ssh config to make it easy to login to the node and to keep the connection alive during some inactivity. To do so, add the following config to your `<HOME_DIR>/.ssh/config`

```
Host embryostage
    HostName <your remote node>
    IdentityFile ~/.ssh/id_rsa
    User <your_remote_username>
    ServerAliveInterval 60
    ServerAliveCountMax 30
```

After this setup, you will be able connect to the node using vscode's `Connect to Host...` command while using the native UI of your computer.

### start VNC server

In the Terminal,
```sh
    vncserver
```
(note the display number :1, :2, ...)


### launch napari from vscode

Setup vscode terminal to use the display created by VNC server.

In the terminal, 
```sh
    export QT_PLUGIN_PATH=$CONDA_PREFIX/plugins # this requires that pyqt is installed in your conda environment.
    export QT_QPA_PLATFORM=xcb
    export DISPLAY =:1 #depending on the number of your vnc display).
```

If you now call `napari` from the terminal, it will launch in the VNC server's display.

### view the display

VNC display is accessible on ports `5901`, `5902`, etc. You can forward this port securely to your computer with vscode using `Ports` tab. 

After you have forwarded the port, you can connect to `localhost:5901` with any vnc client. We recommend tigerVNC for windows and built in Screen Sharing app for Mac.



