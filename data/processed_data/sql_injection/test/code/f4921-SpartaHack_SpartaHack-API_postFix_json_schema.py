<fix/>from marshmallow import Schema,fields,ValidationError</fix>
import ipaddress
import math

"""
schema.dump = used for converting the automap.faqs object to a dictionary good for returning ie cleaning unnecessary fields
schema.validate(request.get_json(force=True)) = used to validate if all the data required for updating and creating the faq is present.4
dump_only = Fields that we need to display when returning the item
load_only = Fields that we need only while dumping from python objects. We use it to stop marshmallow from dumping it while using dump()
"""
<fix/>def ip_test(obj):
    try:
        ipaddress.ip_address(obj)
    except ValueError:
        raise ValidationError('Invalid IPv4 or IPv6 address.') from None</fix>

class Faq_Schema(Schema):
    id = fields.Integer()
    question = fields.String(required=True)
    answer = fields.String(required=True)
    display = fields.Boolean(required=True)
    priority = fields.Integer(required=True)
    placement = fields.String(required=True)
    user_id = fields.Integer()

class Announcement_Schema(Schema):
    id = fields.Integer()
    title = fields.String(required=True)
    description = fields.String(required=True)
    pinned = fields.Boolean(required=True)
    created_at = fields.DateTime(dump_only=True)
    updated_at = fields.DateTime(dump_only=True)

class Hardware_Schema(Schema):
    id = fields.Integer()
    item = fields.String(required=True)
    lender = fields.String(required=True)
    quantity = fields.String(required=True)
    created_at = fields.DateTime(load_only=True)
    updated_at = fields.DateTime(load_only=True)

class Sponsor_Schema(Schema):
    id = fields.Integer()
    name = fields.String(required=True)
    level = fields.String(required=True)
    url = fields.URL(required=True)
    logo_svg_light = fields.String(required=True)
    logo_svg_dark = fields.String()
    logo_png_light = fields.String(required=True)
    logo_png_dark = fields.String()
    created_at = fields.DateTime(load_only=True)
    updated_at = fields.DateTime(load_only=True)

class Schedule_Schema(Schema):
    id = fields.Integer()
    title = fields.String(required=True)
    description = fields.String(required=True)
    time = fields.DateTime(required=True)
    location = fields.String(required=True)
    created_at = fields.DateTime(load_only=True)
    updated_at = fields.DateTime(dump_only=True)

class Application_Schema(Schema):
    id = fields.Integer()
    user_id = fields.Integer(required=True  )
    birth_day = fields.Integer(required=True)
    birth_month = fields.Integer(required=True)
    birth_year = fields.Integer(required=True)
    education = fields.String(required=True)
    university = fields.String()
    other_university = fields.String()
    travel_origin = fields.String()
    graduation_season = fields.String(required=True)
    graduation_year = fields.Integer(required=True)
    major = fields.List(fields.String)
    hackathons = fields.Integer(required=True)
    github = fields.URL()
    linkedin = fields.URL()
    website = fields.URL()
    devpost = fields.URL()
    other_link = fields.URL()
    statement = fields.String()
    created_at = fields.DateTime()
    updated_at = fields.DateTime()
    race = fields.List(fields.String)
    gender = fields.String()
    outside_north_america = fields.String()
    status = fields.String()
    accepted_date = fields.DateTime()

class User_Schema(Schema):
    <fix/>def role_convert(self,obj):
        roles = ["director","judge","mentor","sponsor","organizer","volunteer","hacker"]
        return roles[int(math.log(int(obj.role),2))]

    id = fields.Integer(dump_only=True)
    email = fields.Email(dump_only=True,required=True)</fix>
    encrypted_password = fields.String(required=True)
    <fix/>reset_password_token = fields.String(dump_only=True)</fix>
    reset_password_sent_at = fields.DateTime()
    remember_created_at = fields.DateTime()
    sign_in_count = fields.Integer()
    current_sign_in_at = fields.DateTime()
    last_sign_in_at = fields.DateTime()
    current_sign_in_ip = fields.String(validate=ip_test)
    last_sign_in_ip = fields.String(validate=ip_test)
    <fix/>created_at = fields.DateTime(dump_only=True,required=True)</fix>
    updated_at = fields.DateTime(required=True)
    <fix/>auth_token =fields.String(dump_only=True)</fix>
    confirmation_token = fields.String()
    confirmed_at = fields.DateTime()
    confirmation_sent_at = fields.DateTime()
    <fix/>role = fields.Method("role_convert")
    first_name = fields.String(dump_only=True)
    last_name = fields.String(dump_only=True)
    checked_in = fields.Boolean(dump_only=Tru</fix>e)
