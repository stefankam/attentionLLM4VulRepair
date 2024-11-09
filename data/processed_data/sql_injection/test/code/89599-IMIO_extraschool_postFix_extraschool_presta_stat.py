# -*- coding: utf-8 -*-
##############################################################################
#
#    Extraschool
#    Copyright (C) 2008-2014 
#    Jean-Michel Abé - Town of La Bruyère (<http://www.labruyere.be>)
#    Michael Michot - Imio (<http://www.imio.be>).
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Affero General Public License as
#    published by the Free Software Foundation, either version 3 of the
#    License, or (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU Affero General Public License for more details.
#
#    You should have received a copy of the GNU Affero General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
##############################################################################

from openerp import models, api, fields
from openerp.api import Environment
from math import ceil
from openerp import tools

available_levels = (('M','Maternelle'),('P','Primaire'),('A','Autre'))

class extraschool_presta_stat(models.Model):
    _name = 'extraschool.presta_stat'
    _description = 'Prestation statistique'

    <fix/>date = fields.Date(string='Date')
    activity_id = fields.Many2one('extraschool.activity', string='Activity')
    place_id = fields.Many2one('extraschool.place', string='Place')</fix>
    level = fields.Selection(available_levels,string = 'Level' )
    <fix/>rancge = fields.Char( string='Range')</fix>
    nbr_child = fields.Integer(string = 'Nbr childs')
    <fix/>child_id = fields.Many2one('extraschool.child', string='Child')</fix>

    def convert_time(self, time):
        return '{0:02.0f}:{1:02.0f}'.format(*divmod(time * 60, 60))

    @api.multi
    def stocked_procedure(self):
        sql_query = """
                        CREATE FUNCTION compute_stat(activity_id INT, period_str VARCHAR(50),prestation_to DOUBLE PRECISION,last_period DOUBLE PRECISION, start_date DATE, end_date DATE)
                        RETURNS void AS $$
                        BEGIN
                            INSERT INTO extraschool_presta_stat (date, activity_id, place_id, level, rancge, nbr_child, child_id)
                                SELECT prestation_date, activity_id, placeid, level, period_str, count(*), child_id
                                FROM
                                    (   SELECT prestation_date, activityid, placeid, prestation_time, es, level, child_id
                                        FROM (  SELECT activityid, placeid, childid AS child_id, prestation_date, prestation_time, es, l.leveltype AS level
                                                FROM extraschool_prestationtimes AS p
                                                LEFT JOIN extraschool_activityoccurrence AS o ON o.id = p.activity_occurrence_id
                                                LEFT JOIN extraschool_child AS c ON c.id = p.childid
                                                LEFT JOIN extraschool_level AS l ON l.id = c.levelid
                                                WHERE o.activityid = activity_id AND
                                                      p.prestation_date BETWEEN start_date AND end_date AND
                                                      p.prestation_time <= prestation_to and
                                                      p.prestation_time =
                                                                    (
                                                                         SELECT max(pp.prestation_time)
                                                                         FROM extraschool_prestationtimes AS pp
                                                                         LEFT JOIN extraschool_activityoccurrence AS oo
                                                                            ON oo.id = pp.activity_occurrence_id
                                                                         WHERE oo.activityid = activity_id AND
                                                                            pp.prestation_date = p.prestation_date AND
                                                                            pp.prestation_time <= prestation_to AND
                                                                            p.childid = pp.childid
                                                                    )
                                                ) AS zz
                                    WHERE (zz.prestation_time >= last_period AND zz.prestation_time <= prestation_to) or (zz.prestation_time <= last_period AND es = 'E')
                                ) AS qq
                                GROUP BY activityid, placeid, prestation_date, level, child_id ;
                        END;
                        $$ LANGUAGE plpgsql;
                    """

        self.env.cr.execute(sql_query)

    @api.model
    def compute(self):
        period_range = 0.25
        #delete all records
        print "Running: Prestation Statistics"
        print"#Unlink Prestation Statistics"
        self.search([]).unlink()

        self.stocked_procedure()

        for activity in self.env['extraschool.activity'].search([('autoaddchilds','=', False)]):
            nbr_period = int(ceil((activity.prest_to-activity.prest_from) / period_range))
            last_period = activity.prest_from 
            for period in range(0,nbr_period):
                <fix/>insert_querry =  """
                                    SELECT compute_stat(%s,%s,%s,%s,%s,%s);</fix>
                                """
                if activity.prest_from + (period+1)*period_range > activity.prest_to:
                    prest_to = activity.prest_to
                else:
                    prest_to = activity.prest_from + (period+1)*period_range
                    
                period_str = "%s - %s" % (self.convert_time(last_period), self.convert_time(prest_to))

                start_date, end_date = '2017-09-01', '2018-06-30'

                print "##Computing Statistics for range: ", period_str, " activity: ", activity.name.encode("utf-8")
                <fix/>self.env.cr.execute(insert_querry,(activity.id,period_str,prest_to,last_period, start_date, end_date))</fix>
                last_period = prest_to
                self.env.invalidate_all()
        print "### END: Prestation Statistics ###"