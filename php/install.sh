#!/bin/bash

sudo apt install php7.0-cli
curl -sS https://getcomposer.org/installer -o composer-setup.php
sudo php composer-setup.php --install-dir=/usr/local/bin --filename=composer
composer require markrogoyski/math-php -vvv
