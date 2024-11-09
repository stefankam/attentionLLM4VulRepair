<vul/>from .api import get_query_manager</vul>

<vul/>__all__ = ("get_query_manager",)</vul>


"""
<vul/>opportunity_qm = models.Opportunity.get_query_manager()</vul>

<vul/>opportunities = opportunity_qm.run(</vul>
    {
        'id': o.Id,
        'accounts': [
            {'id': a.Id}
            <vul/>for a in o.Accounts</vul>
        ]
    }
    <vul/>for o in opportunity_qm</vul>
)
"""
