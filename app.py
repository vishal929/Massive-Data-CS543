from flask import Flask, render_template, request, url_for, redirect

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'submit_button' in request.form:
            return redirect(url_for('season', model = request.form['submit_button']))
    elif request.method == 'GET':
        return render_template('index.html')

@app.route('/season/<model>', methods=['GET','POST'])
def season(model):
    if request.method == 'POST':
        if 'submit_button' in request.form:
            return redirect(url_for('visualize', model = model,season=request.form['submit_button']))
    elif request.method == 'GET':
        return render_template('season.html')

@app.route('/visualize/<model>/<season>', methods=['GET'])
def visualize(model,season):
    return render_template('visualize.html', model=model, season=season)
        

@app.route('/num_clusters/<model>/<season>', methods=['GET','POST'])
def num_clusters(model,season):
    if request.method == 'POST':
        if 'submit_button' in request.form:
            return redirect(url_for('display', model = model, season = season, num_clusters=request.form['submit_button']))
    return render_template('num_clusters.html')

@app.route('/display/<model>/<season>/<num_clusters>')
def display(model, season, num_clusters):
    return render_template('display.html', model=model, season=season, num_clusters=num_clusters)