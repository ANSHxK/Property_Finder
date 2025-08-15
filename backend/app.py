from flask import Flask, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
import os

app = Flask(__name__)

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///site.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
migrate = Migrate(app, db)

# Define models
class Property(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    location = db.Column(db.String(100), nullable=False)
    price = db.Column(db.Float, nullable=False)

# Sample route for getting properties
@app.route('/api/properties', methods=['GET'])
def get_properties():
    properties = Property.query.all()
    return jsonify([{'id': p.id, 'name': p.name, 'location': p.location, 'price': p.price} for p in properties])

# Sample route for adding a property
@app.route('/api/properties', methods=['POST'])
def add_property():
    # In a real application, you'd get data from request JSON
    new_property = Property(name="Sample Property", location="Sample Location", price=100000)
    db.session.add(new_property)
    db.session.commit()
    return jsonify({'message': 'Property added!'}), 201

# Error handling
@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Not found'}), 404

if __name__ == '__main__':
    app.run(debug=True)