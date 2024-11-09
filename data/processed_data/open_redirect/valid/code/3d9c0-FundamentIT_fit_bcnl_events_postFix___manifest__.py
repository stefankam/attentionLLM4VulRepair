# -*- coding: utf-8 -*-
# Copyright 2018 Fundament IT
# License AGPL-3.0 or later (http://www.gnu.org/licenses/agpl).
{
    'name': 'FIT BCNL Events',
    'category': 'Website',
    <fix/>'version': '10.0.1.0.0',</fix>
    'author': 'Fundament IT',
    'website': 'https://fundament.it/',
    'licence': 'AGPL-3',
    'depends': ['website_event'],
    'summary':"""""",
    'description': """
Extension for the default Events module; used for BCNL, .
    """,
    'data': [
        'data/fit_actions.xml',
        'data/fit_event_views.xml',
        'data/fit_product_views.xml',
        'data/fit_res_partner_view.xml',
        'report/fit_report_subscription_view.xml',
        'security/user_groups.xml',
        'views/fit_event_views.xml',
        'views/fit_subscription_templates.xml',
        'views/fit_website_event_sale_templates.xml',
        'views/fit_website_event_templates.xml',
        'views/fit_website_templates.xml',
    ],
    'installable': True,
}