<vul/>from marshmallow import Schema,fields</vul>
import ipaddress

"""
schema.dump = used for converting the automap.faqs object to a dictionary good for returning ie cleaning unnecessary fields
schema.validate(request.get_json(force=True)) = used to validate if all the data required for updating and creating the faq is present.4
dump_only = Fields that we need to display when returning the item
load_only = Fields that we need only while dumping from python objects. We use it to stop marshmallow from dumping it while using dump()
"""
<vul/>def ip_test(ip):
        try:
            ipaddress.ip_address(ip)
            return True
        except ValueError:
            return False</vul>

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
    <vul/>id = fields.Integer()
    email = fields.Email(required=True)</vul>
    encrypted_password = fields.String(required=True)
    <vul/>reset_password_token = fields.String()</vul>
    reset_password_sent_at = fields.DateTime()
    remember_created_at = fields.DateTime()
    sign_in_count = fields.Integer()
    current_sign_in_at = fields.DateTime()
    last_sign_in_at = fields.DateTime()
    current_sign_in_ip = fields.String(validate=ip_test)
    last_sign_in_ip = fields.String(validate=ip_test)
    <vul/>created_at = fields.DateTime(required=True)</vul>
    updated_at = fields.DateTime(required=True)
    <vul/>auth_token =fields.String()</vul>
    confirmation_token = fields.String()
    confirmed_at = fields.DateTime()
    confirmation_sent_at = fields.DateTime()
    role = fields.Integer()
    first_name = fields.String()
    last_name = fields.String()
    checked_in = fields.Boolean(required=True)