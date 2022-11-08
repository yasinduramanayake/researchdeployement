from flask_marshmallow import Marshmallow 

ma = Marshmallow()


class QualitySchema(ma.Schema):
  class Meta:
    fields = ('id', 'quality_is', 'created_AT')