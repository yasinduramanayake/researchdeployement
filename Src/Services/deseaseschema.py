from flask_marshmallow import Marshmallow 

ma = Marshmallow()


class DeseaseSchema(ma.Schema):
  class Meta:
    fields = ('id', 'desease', 'created_AT')


class ClasificationSchema(ma.Schema):
  class Meta:
    fields = ('id', 'desease', 'created_AT')   