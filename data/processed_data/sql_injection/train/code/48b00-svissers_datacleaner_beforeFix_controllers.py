from flask import (
    Blueprint,
    request,
    jsonify,
    redirect,
    url_for,
    render_template,
    flash,
    Response
)
from flask_login import login_required, current_user
from app.Data.operations import create_action, get_dataset_with_id
<vul/>from app.Data.helpers import table_name_to_object</vul>
from app.Data.Transform.operations import (
    restore_original,
    change_attribute_type,
    delete_rows,
    fill_null_with,
    fill_null_with_average,
    fill_null_with_median,
    rename_attribute,
    delete_attribute,
    one_hot_encode,
    normalize_attribute,
    discretize_width,
    discretize_eq_freq,
    find_replace,
    regex_find_replace,
    substring_find_replace
)

_transform = Blueprint('transform_bp', __name__, url_prefix='/data/transform')


@_transform.route('/rename_column', methods=['POST'])
@login_required
def rename_column():
    dataset = get_dataset_with_id(request.args.get('dataset_id'))
    col = request.form['column']
    new_name = request.form['new_name']
    try:
        rename_attribute(dataset.working_copy, col, new_name)
        create_action(
            'Renamed column {0} to {1}'.format(col, new_name),
            dataset.id,
            current_user.id
        )
    except:
        flash('An unexpected error occured while renaming the column', 'danger')
    else:
        flash('Column renamed successfully.', 'success')

    return redirect(request.referrer)


@_transform.route('/delete_column', methods=['POST'])
@login_required
def delete_column():
    dataset = get_dataset_with_id(request.args.get('dataset_id'))
    col = request.form['column']
    try:
        delete_attribute(dataset.working_copy, col)
        create_action(
            'Deleted column {0}'.format(col),
            dataset.id,
            current_user.id
        )
    except:
        flash('An unexpected error occured while deleting the column', 'danger')
    else:
        flash('Column deleted successfully.', 'success')

    return redirect(request.referrer)


@_transform.route('one_hot_encode_column', methods=['POST'])
@login_required
def one_hot_encode_column():
    dataset = get_dataset_with_id(request.args.get('dataset_id'))
    col = request.form['column']
    try:
        one_hot_encode(dataset.working_copy, col)
        create_action(
            'One-hot-encoded {0}'.format(col),
            dataset.id,
            current_user.id
        )
    except:
        flash('An unexpected error occured while one-hot-encoding the column',
              'danger'
              )
    else:
        flash('Column one-hot-encoded successfully.', 'success')

    return redirect(request.referrer)


@_transform.route('normalize_column', methods=['POST'])
@login_required
def normalize_column():
    dataset = get_dataset_with_id(request.args.get('dataset_id'))
    col = request.form['column']
    try:
        normalize_attribute(dataset.working_copy, col)
        create_action(
            'Normalized {0}'.format(col),
            dataset.id,
            current_user.id
        )
    except:
        flash('An unexpected error occured while normalizing the column',
              'danger'
              )
    else:
        flash('Column normalized successfully.', 'success')

    return redirect(request.referrer)


@_transform.route('/discretize_column', methods=['POST'])
@login_required
def discretize_column():
    dataset = get_dataset_with_id(request.args.get('dataset_id'))
    column = request.form['column']
    intervals = request.form['intervals']

    try:
        if intervals == 'equal-distance':
            amount = request.form['amount-dist']
            discretize_width(dataset.working_copy, column, int(amount))
        elif intervals == 'equal-frequency':
            amount = request.form['amount-freq']
            discretize_eq_freq(dataset.working_copy, column, int(amount))
        else:
            edges = str(request.form['custom-edges'])
            edges = edges.replace(' ', '')
            edge_list = edges.split(',')
            if len(edge_list) < 2:
                raise ValueError
            for i in range(len(edge_list)):
                edge_list[i] = float(edge_list[i])
            discretize_width(dataset.working_copy, column, edge_list)

    except ValueError:
        flash('Invalid list of edges provided.',
              'danger'
              )
    except:
        flash('An unexpected error occured while discretizing the column',
              'danger'
              )
    else:
        flash('Column discretized successfully.', 'success')

    return redirect(request.referrer)


@_transform.route('/delete_selection', methods=['POST'])
@login_required
def delete_selection():
    dataset = get_dataset_with_id(request.args.get('dataset_id'))
    selected_data = request.form.getlist("data_id")
    table = table_name_to_object(dataset.working_copy)
    for data in selected_data:
        table.delete(table.c.index == data).execute()
    create_action(
        'deleted selected items',
        dataset.id,
        current_user.id
    )
    return redirect(request.referrer)


@_transform.route('/delete_predicate', methods=['POST'])
@login_required
def delete_predicate():
    dataset = get_dataset_with_id(request.args.get('dataset_id'))
    table = table_name_to_object(dataset.working_copy)
    condition = ''
    columns = []
    conditions = []
    operators = []
    logics = []
    for i in request.form:
        if i.startswith('column'):
            columns.append(i)
        elif i.startswith('condition'):
            conditions.append(i)
        elif i.startswith('logical'):
            logics.append(i)
        elif i.startswith('operator'):
            operators.append(i)
    columns.sort()
    conditions.sort()
    logics.sort()
    operators.sort()
    for i in range(len(columns)):
        if i != len(columns) - 1:
            condition += '"' + request.form[columns[i + 1]] + '"'
            if request.form[operators[i + 1]] == 'CONTAINS':
                condition += ' ~ '
            elif request.form[operators[i + 1]] == 'NOT CONTIANS':
                condition += ' !~ '
            else:
                condition += request.form[operators[i + 1]]
            condition += '\'' + request.form[conditions[i + 1]] + '\''
            condition += ' ' + request.form[logics[i]] + ' '
        else:
            condition += '"' + request.form[columns[0]] + '"'
            if request.form[operators[0]] == 'CONTAINS':
                condition += ' ~ '
            elif request.form[operators[0]] == 'NOT CONTIANS':
                condition += ' !~ '
            else:
                condition += request.form[operators[0]]
            <vul/>condition += '\'' + request.form[conditions[0]] + '\''</vul>

    try:
        <vul/>delete_rows(table.name, condition)
        create_action('rows deleted with condition "{0}"'
                      .format(condition), dataset.id, current_user.id
                      )
    except:</vul>
        flash('condition "{0}" not valid'.format(condition), 'danger')
    <vul/>else:
        flash('successfully deleted rows using condition "{0}"'
              .format(condition), 'success'
              )</vul>
    return redirect(request.referrer)


@_transform.route('/reset', methods=['GET'])
@login_required
def reset():
    dataset = get_dataset_with_id(request.args.get('dataset_id'))
    restore_original(dataset.working_copy)
    create_action(
        'restored dataset to original state',
        dataset.id,
        current_user.id
    )
    return redirect(request.referrer)


@_transform.route('/change_type', methods=['POST'])
@login_required
def change_type():
    dataset = get_dataset_with_id(request.args.get('dataset_id'))
    table = table_name_to_object(dataset.working_copy)
    col = request.form['column']
    col = col[:col.find('(')-1]
    new_type = request.form['type']
    if col != '' and new_type != '':
        try:
            change_attribute_type(table.name, col, new_type)
            create_action('type {0} changed to {1}'.format(col, new_type), dataset.id, current_user.id)
        except:
            flash('{0} could not be converted to {1}'.format(col, new_type), 'danger')
        else:
            flash('{0} successfully  converted to {1}'.format(col, new_type), 'success')

    return redirect(request.referrer)


@_transform.route('/find_and_replace', methods=['POST'])
@login_required
def find_and_replace():
    dataset = get_dataset_with_id(request.args.get('dataset_id'))
    col = request.form['column']
    find = request.form['find']
    match_mode = request.form['match-mode']
    replace = request.form['replace']

    if match_mode == 'full-match':
        find_replace(dataset.working_copy, col, find, replace)
    elif match_mode == 'substring-match':
        replace_mode = request.form['replace-mode']
        if replace_mode == 'full-replace':
            substring_find_replace(dataset.working_copy,
                                   col,
                                   find,
                                   replace,
                                   full=True)
        elif replace_mode == 'substring-replace':
            substring_find_replace(dataset.working_copy,
                                   col,
                                   find,
                                   replace,
                                   full=False)
    elif match_mode == 'regex-match':
        regex_find_replace(dataset.working_copy, col, find, replace)

    return redirect(request.referrer)


@_transform.route('/fill_null', methods=['POST'])
@login_required
def fill_null():
    dataset = get_dataset_with_id(request.args.get('dataset_id'))
    column_and_type = request.form['column']
    column_name = column_and_type[:column_and_type.find(' ')]
    column_type = column_and_type[column_and_type.find('(')+1:column_and_type.rfind(')')]
    fill_value = request.form['fill_value']

    try:
        if fill_value == '~option-average~':
            if column_type not in ['INTEGER', 'BIGINT', 'DOUBLE PRECISION']:
                flash('Operation not supported for this column type.', 'danger')
            else:
                fill_null_with_average(dataset.working_copy, column_name)
                create_action(
                    'Filled null values in {0} with average'.format(column_name),
                    dataset.id,
                    current_user.id
                )
        elif fill_value == '~option-median~':
            if column_type not in ['INTEGER', 'BIGINT', 'DOUBLE PRECISION']:
                flash('Operation not supported for this column type.', 'danger')
            else:
                fill_null_with_median(dataset.working_copy, column_name)
                create_action(
                    'Filled null values in {0} with median'.format(column_name),
                    dataset.id,
                    current_user.id
                )
        else:
            is_text_type = column_type in ['TEXT',
                                           'VARCHAR(10)',
                                           'VARCHAR(25)',
                                           'VARCHAR(255)']
            fill_null_with(
                dataset.working_copy,
                column_name,
                fill_value,
                is_text_type
            )
            create_action(
                'Filled null values in {0} with {1}'
                .format(column_name, fill_value),
                dataset.id,
                current_user.id
            )
    except:
        flash(
            'An unexpected error occured while performing the operation',
            'danger'
            )
    else:
        flash('Fill operation completed successfully', 'success')

    return redirect(request.referrer)
