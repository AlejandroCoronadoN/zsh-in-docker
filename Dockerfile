# This is only used for developing the zsh-in-docker script, but can be used as an example.

FROM continuumio/anaconda3
WORKDIR /

ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd -s /bin/bash --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo wget \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    &&  sudo -u $USERNAME  -i \
    # Clean up
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*


# RUN sh -c "$(wget -O- https://github.com/deluan/zsh-in-docker/releases/download/v1.1.1/zsh-in-docker.sh)" -- \
COPY zsh-in-docker.sh /tmp
RUN /tmp/zsh-in-docker.sh \
    -t https://github.com/denysdovhan/spaceship-prompt \
    -a 'SPACESHIP_PROMPT_ADD_NEWLINE="false"' \
    -a 'SPACESHIP_PROMPT_SEPARATE_LINE="false"' \
    -p git \
    -p https://github.com/zsh-users/zsh-autosuggestions \
    -p https://github.com/zsh-users/zsh-completions \
    -p https://github.com/zsh-users/zsh-history-substring-search \
    -p https://github.com/zsh-users/zsh-syntax-highlighting \
    -p 'history-substring-search' \
    #-p https://github.com/ohmyzsh/ohmyzsh/tree/master/plugins/dirhistory \
    #-p https://github.com/ohmyzsh/ohmyzsh/tree/master/plugins/sudo \
    #-p https://github.com/ohmyzsh/ohmyzsh/tree/master/plugins/web-search \
    -p  copydir \
    -p copyfile \
    -p copybuffer \
    -p jsontools \
    -p history \
    -a 'bindkey "\$terminfo[kcuu1]" history-substring-search-up' \
    -a 'bindkey "\$terminfo[kcud1]" history-substring-search-down'

ENTRYPOINT [ "/bin/zsh" ]
CMD ["-l"]

RUN ls
RUN pwd
#RUN sudo chown -R username /opt/conda/
COPY /condaenvs/ /condaenvs/
#RUN sudo chown -R +x /opt/onda

RUN echo "unset $USER_UID $USER_GID $USERNAME"  >> ~/.zshrc
RUN echo "unset $USER_UID $USER_GID $USERNAME"  >> ~/.bashrc

RUN conda env create -f /condaenvs/environment_cpu.yml\
 && . "$(conda info --base)/etc/profile.d/conda.sh" 