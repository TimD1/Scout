# Scout

A basic CNN which can be trained to detect specific types of positions (most commonly, candidate variants) when given a read pileup.


## Developer Quickstart

$ git clone https://github.com/TimD1/Scout.git
$ cd Scout
$ python3 -m venv venv3
$ source venv3/bin/activate
(venv3) $ pip install --upgrade pip
(venv3) $ pip install -r requirements.txt
(venv3) $ python setup.py develop
(venv3) $ scout --help
