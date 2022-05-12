from flask import Flask, render_template, request, make_response, redirect, flash, url_for
import depth_table
import pandas as pd
app = Flask(__name__)

app.config['SECRET_KEY'] = 'mysecretkey'


posts = [{'title': 'Welcome to Alpha Tau Treatment GridTracker',
           'info': 'Send live data directly to the surgeon!',
           }]

@app.route("/", methods=['GET', 'POST'])
@app.route("/home", methods=['GET', 'POST'])
def home():
    return render_template("home.html", posts=posts)


@app.route("/data", methods=['GET', 'POST'])
def data():
    if request.method == 'POST':
        depth_table.mark_grid()
        # f = request.form.get('csvfile')
        # data = []
        # with open(f) as file:
        #     csvfile = csv.reader(file)
        #     for row in csvfile:
        #         data.append(row)
        # data = pd.DataFrame(data)
        # return render_template("data.html", data=data.to_html(header='False', index=False))
        return redirect(request.url)
    if request.method == 'GET':
        data = pd.read_csv("static/items_example.csv")
        data["Number of Seeds"] = data["Number of Seeds"].fillna(0).astype(int)
        body = data.to_string(header=False,
                          index=False,
                          index_names=False)
        body_no_spaces = ''
        for line in body.split('\n'):
            body_no_spaces = body_no_spaces + ','.join(line.split()) + '\n'
        headers = ','.join(data.columns.to_list())
        str_data = headers + '\n' + body_no_spaces.replace('NaN', '')
        response = make_response(str_data)
        cd = 'attachment; filename=mycsv.csv'
        response.headers['Content-Disposition'] = cd
        response.mimetype = 'text/csv'
        return response

@app.route("/map")
def display_map():
    return redirect(url_for('static', filename='img/map.png'))


def run():
    app.run(host="0.0.0.0", port=5000, debug=True)


if __name__ == "__main__":
    run()

