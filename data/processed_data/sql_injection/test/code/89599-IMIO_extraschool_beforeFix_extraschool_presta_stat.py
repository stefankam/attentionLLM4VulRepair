<vul/></vul># -*- coding: utf-8 -*-
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

available_levels = (('M','Maternelle'),('P','Primaire'),('A','Autre'))

class extraschool_presta_stat(models.Model):
    _name = 'extraschool.presta_stat'
    _description = 'Prestation statistique'

    <vul/>date = fields.Date(string='Date', index=True)
    activity_id = fields.Many2one('extraschool.activity', string='Activity', index=True)
    place_id = fields.Many2one('extraschool.place', string='Place', index=True)</vul>
    level = fields.Selection(available_levels,string = 'Level' )
    <vul/>rancge = fields.Char( string='Range', index=True)</vul>
    nbr_child = fields.Integer(string = 'Nbr childs')
    <vul/>child_id = fields.Many2one('extraschool.child', string='Child', index=True)</vul>

    def convert_time(self, time):
        return '{0:02.0f}:{1:02.0f}'.format(*divmod(time * 60, 60))

    @api.model
    def compute(self):
        period_range = 0.25
        #delete all records
        print "Running: Prestation Statistics"
        print"#Unlink Prestation Statistics"
        self.search([]).unlink()
        
        for activity in self.env['extraschool.activity'].search([('autoaddchilds','=', False)]):
            nbr_period = int(ceil((activity.prest_to-activity.prest_from) / period_range))
            last_period = activity.prest_from 
            for period in range(0,nbr_period):
                <vul/>insert_querry = """
                                insert into extraschool_presta_stat
                                    (date, activity_id, place_id, level, rancge, nbr_child, child_id)                                    
                                    select prestation_date, %s, placeid, level, %s, count(*), child_id
                                    from 
                                    (select prestation_date, activityid, placeid, prestation_time, es, level, child_id
                                        from
                                        (select activityid, placeid, childid AS child_id, prestation_date, prestation_time, es, l.leveltype as level
                                        from extraschool_prestationtimes p
                                        left join extraschool_activityoccurrence o on o.id = p.activity_occurrence_id 
                                        left join extraschool_child c on c.id = p.childid
                                        left join extraschool_level l on l.id = c.levelid
                                        where o.activityid = %s and                                                
                                              p.prestation_time <= %s and
                                              p.prestation_time =
                                            (
                                              select max(pp.prestation_time)
                                              from extraschool_prestationtimes pp
                                              left join extraschool_activityoccurrence oo 
                                                on oo.id = pp.activity_occurrence_id
                                              where oo.activityid = %s and
                                                pp.prestation_date = p.prestation_date and
                                                pp.prestation_time <= %s and
                                                p.childid = pp.childid        
                                            )
                                        ) zz
                                        where (zz.prestation_time >= %s and zz.prestation_time <= %s) or
                                            (zz.prestation_time <= %s and es = 'E')
                                    ) qq
                                    group by activityid, placeid, prestation_date, level, child_id</vul>
                                """
                if activity.prest_from + (period+1)*period_range > activity.prest_to:
                    prest_to = activity.prest_to
                else:
                    prest_to = activity.prest_from + (period+1)*period_range
                    
                period_str = "%s - %s" % (self.convert_time(last_period), self.convert_time(prest_to))

                print "##Computing Statistics for range: ", period_str, " activity: ", activity.name.encode("utf-8")
                <vul/>self.env.cr.execute(insert_querry,(activity.id,period_str,activity.id,prest_to,activity.id,prest_to,last_period,prest_to,last_period))</vul>
                last_period = prest_to
                self.env.invalidate_all()
        print "### END: Prestation Statistics ###"