from flask import Flask, render_template, request, url_for, redirect, send_file

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'submit_button' in request.form:
            if request.form['submit_button'] == "CURE Analysis":
                return redirect(url_for('season'))
            if request.form['submit_button'] == 'Deep Learning':
                return redirect(url_for('deep'))
    elif request.method == 'GET':
        return render_template('index.html')

@app.route('/cure_analysis/<season>', methods=['GET'])
def cure_analysis(season):
    if season == 'Winter':
        return render_template('winter_analysis.html')
    if season == 'Spring':
        return render_template('spring_analysis.html')
    if season == 'Summer':
        return render_template('summer_analysis.html')
    if season == 'Fall':
        return render_template('fall_analysis.html')
    if season == 'Cumulative':
        return render_template('cumulative_analysis.html')

@app.route('/season', methods=['GET','POST'])
def season():
    if request.method == 'POST':
        if 'submit_button' in request.form:
            return redirect(url_for('cure_analysis', season=request.form['submit_button']))
    elif request.method == 'GET':
        return render_template('season.html')

@app.route('/deep', methods=['GET','POST'])
def deep():
    if request.method == 'POST':
        if 'submit_button' in request.form:
            return redirect(url_for('download', model=request.form['submit_button']))
    elif request.method == 'GET':
        return render_template('deep.html')

@app.route('/download/<model>', methods=['GET'])
def download(model):
    if model == 'Categorical':
        return send_file('.\static\DeepLearning/categorical_models.zip', as_attachment=True)
    elif model == 'Regression':
        return send_file('.\static\DeepLearning/regression_models.zip', as_attachment=True)
    else:
        return redirect(url_for('deep'))

# @app.route('/visualize/<model>/<season>', methods=['GET'])
# def visualize(model,season):
#     return render_template('visualize.html', model=model, season=season)
        

# @app.route('/num_clusters/<model>/<season>', methods=['GET','POST'])
# def num_clusters(model,season):
#     if request.method == 'POST':
#         if 'submit_button' in request.form:
#             return redirect(url_for('display', model = model, season = season, num_clusters=request.form['submit_button']))
#     return render_template('num_clusters.html')

# @app.route('/display/<model>/<season>/<num_clusters>')
# def display(model, season, num_clusters):
#     return render_template('display.html', model=model, season=season, num_clusters=num_clusters)