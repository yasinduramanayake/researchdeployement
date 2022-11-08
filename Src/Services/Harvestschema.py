from flask_marshmallow import Marshmallow 

ma = Marshmallow()


class ProductSchema(ma.Schema):
  class Meta:
    fields = ('id', 'full_harvest', 'can_be_utilized','Created_At')