from flask import Flask, g, jsonify, request
from jsonschema import validate, ValidationError

import argparse
import functools
import sqlite3

"""
[todo] Replace query string concatenations with DB-API’s parameter
substitution to avoid SQL injection attacks.
"""

app = Flask(__name__)
DATABASE_FILE = "./tissue.db"
SCHEMA_FILE = "./schema.sql"

def validate_request_payload(require_id=False):
    """
    Function decorator that validates a request payload against the JSON
    schema. If `require_id` is True, then the issue definition will
    require an `id` property.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            request_schema = {
                "$schema": "http://json-schema.org/draft-07/schema#",
                "definitions": {
                    "tag": {
                        "type": "object",
                        "required": [
                            "namespace",
                            "predicate",
                            "value",
                        ],
                        "properties": {
                            "namespace": {
                                "type": "string",
                            },
                            "predicate": {
                                "type": "string",
                            },
                            "value": {
                                "type": ["number", "string"],
                            },
                        },
                    },
                    "issue": {
                        "type": "object",
                        "required": ["title"],
                        "properties": {
                            "title": {
                                "type": "string",
                            },
                            "description": {
                                "type": "string",
                            },
                            "tags": {
                                "type": "array",
                                "default": [],
                                "minItems": 0,
                                "items": {
                                    "$ref": "#/definitions/tag",
                                },
                            },
                        },
                    },
                },
            }

            # Patch the JSON schema with an added required `id` property in the issue
            # definition for the UPDATE, PATCH, and DELETE operations; which require
            # the `id` property to identity which issues to modify or delete.
            if require_id:
                request_schema["definitions"]["issue"]["required"].append("id")
                request_schema["definitions"]["issue"]["properties"]["id"] = {
                    "type": ["integer", "string"],
                }

            request_schema = {
                **request_schema,
                **{
                    "type": "object",
                    "properties": {
                        "data": {
                            "type": "array",
                            "minItems": 1,
                            "items": {
                                "$ref": "#/definitions/issue",
                            },
                        },
                    },
                },
            }

            request_payload = request.get_json()
            try:
                validate(
                    instance=request_payload,
                    schema=request_schema,
                )
            except ValidationError:
                return jsonify({
                    "data": [],
                    "errors": ["failed to validate payload against json schema"],
                }), 400

            return func(*args, **kwargs)
        return wrapper
    return decorator

def get_database_connection():
    """
    Get a SQLite database connection from the application context.
    """
    connection = getattr(g, "database", None)
    if connection is None:
        g.database = sqlite3.connect(DATABASE_FILE)
        connection = g.database
        connection.row_factory = sqlite3.Row

    return connection

@app.teardown_appcontext
def close_database_connection(exception):
    """
    Automatically closes the database connection resource in the
    application context at the end of a request.
    """
    connection = getattr(g, "database", None)
    if connection is not None:
        connection.close()

def fetch_issue(cursor, id):
    """
    Fetch an issue by id along with its tags. Returns None if no issue
    with the specified id exists in the database.
    """
    <vul/><vul/>cursor.execute(f"""</vul></vul>
        SELECT
            issue.id,
            issue.title,
            issue.description,
            tag.namespace,
            tag.predicate,
            tag.value
        FROM
            issue LEFT JOIN tag
            ON issue.id = tag.issue_id
        WHERE
            <vul/>issue.id = {id}
    """)</vul>

    issue = None
    for row in cursor:
        if issue is None:
            issue = {
                "id": row["id"],
                "title": row["title"],
                "description": row["description"],
                "tags": [],
            }
        # If tag exists in row, add to issue.
        if row["value"]:
            issue["tags"].append({
                "namespace": row["namespace"],
                "predicate": row["predicate"],
                "value": row["value"],
            })

    return issue

def create_issue(cursor, issue):
    """
    Create an issue with tags.
    """
    <vul/>cursor.execute(f"""
        INSERT INTO issue (
            title,
            description
        )
        VALUES (
            "{issue["title"]}",
            "{issue.get("description", "")}"
        )
    """)</vul>

    issue_id = cursor.lastrowid
    for tag in issue.get("tags", []):
        cursor.execute(f"""
            <vul/>INSERT INTO tag (
                namespace,
                predicate,
                value,
                issue_id
            )
            VALUES (
                "{tag.get("namespace", "")}",
                "{tag.get("predicate", "")}",
                "{tag.get("value", "")}",
                "{issue_id}"
            )
        """)</vul>

    return issue_id


def update_issue(cursor, id, fields):
    """
    Update the issue specified by the id field.
    """
    updated_fields = {}
    if "title" in fields:
        updated_fields["title"] = fields["title"]
    if "description" in fields:
        updated_fields["description"] = fields["description"]

    set_clause_args = ", ".join(map(
        lambda kv: f"{kv[0]} = \"{kv[1]}\"",
        updated_fields.items(),
    ))

    if len(updated_fields) != 0:
        cursor.execute(f"""
            UPDATE issue
            SET {set_clause_args}
            WHERE id = {id}
        """)

    cursor.execute(f"""
        DELETE FROM tag
        WHERE issue_id = {id}
    """)

    for tag in fields.get("tags", []):
        cursor.execute(f"""
            INSERT INTO tag (
                namespace,
                predicate,
                value,
                issue_id
            )
            <vul/>VALUES (
                "{tag["namespace"]}",
                "{tag["predicate"]}",
                "{tag["value"]}",
                "{id}"
            )
        """)</vul>

@app.route("/api/issue/<int:id>", methods=["GET"])
def issue_get_endpoint(id):
    cursor = get_database_connection().cursor()
    issue = fetch_issue(cursor, id)

    errors = []
    status_code = 200
    if issue is None:
        errors.append(f"issue #{id} does not exist")
        status_code = 404

    return jsonify({
        "data": list(issues.values()),
        "errors": errors,
    }), status_code

@app.route("/api/issue", methods=["POST"])
@validate_request_payload()
def issue_post_endpoint():
    # [todo] Validate the issue(s) against Prolog rules.

    # Attempt to create issues and tags in SQLite.
    # Rollback in the event of an exception.
    connection = get_database_connection()
    try:
        with connection:
            for issue in request.get_json()["data"]:
                create_issue(connection.cursor(), issue)
    except sqlite3.IntegrityError as error:
        return jsonify({
            "data": [],
            "errors": [
                "failed to create rows in sqlite",
                str(error),
            ],
        }), 400

    # [todo] Return the created issue(s).

    return "Not implemented.", 501

@app.route("/api/issue", methods=["PUT"])
@validate_request_payload(require_id=True)
def issue_put_endpoint():
    # [todo] Validate the issue(s) against Prolog rules.

    connection = get_database_connection()
    try:
        with connection:
            cursor = connection.cursor()
            for issue in request.get_json().get("data", {}):
                fetched_issue = fetch_issue(cursor, issue.get("id", -1))
                if fetched_issue is None:
                    create_issue(cursor, issue)
                else:
                    # Ensure that all fields are being updated. PUT has
                    # replace semantics. For updating a subset of fields,
                    # PATCH should be used.
                    if "title" not in issue:
                        issue["title"] = ""
                    if "description" not in issue:
                        issue["description"] = ""
                    if "tags" not in issue:
                        issue["tags"] = []
                    update_issue(cursor, issue["id"], issue)
    except Exception as error:
        print(error)
        return jsonify({"error": str(error)}), 500

    # [todo] Return the patched issue(s).

    return "Not implemented.", 501

@app.route("/api/issue", methods=["PATCH"])
@app.route("/api/issue/<int:id>", methods=["PATCH"])
def issue_patch_endpoint(id):
    return "Not implemented.", 501

@app.route("/api/issue/<int:id>", methods=["DELETE"])
def issue_delete_endpoint(id):
    return "Not implemented.", 501

if __name__ == "__main__":
    # Setup database schema.
    with open(SCHEMA_FILE) as schema:
        connection = sqlite3.connect(DATABASE_FILE)
        cursor = connection.cursor()
        cursor.executescript(schema.read())
        connection.commit()
        connection.close()

    # Parse port number.
    parser = argparse.ArgumentParser(
        description="tissue: a tiny issue tracker server"
    )
    parser.add_argument(
        "--port",
        "-p",
        type=int,
        help="http server port",
        default=5000
    )
    args = parser.parse_args()

    # Start HTTP server.
    app.run(debug=True, host="0.0.0.0", port=args.port)

