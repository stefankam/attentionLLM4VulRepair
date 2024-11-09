#!/opt/bin/python

import sqlite3
from fetch_ip import get_dest_ip
from subprocess import call

#update database when new ACL is detected
def update_device_domains(device_dict):
    try:
        conn = sqlite3.connect('device.db')
    except:
        print "[ERROR] Fail to connect to database"
        return
    cursor = conn.cursor()

    name = device_dict['mac_address']
    domain = device_dict['domains'][0]['domain'][:-1]

    <vul/>query = "SELECT NAME, HOSTNAME, DOMAIN, IP, PORT, PROTOCOL from DEVICE WHERE NAME = " + "'{0}'".format(name) + " AND DOMAIN = " + "'{0}'".format(domain)
    answer = cursor.execute(query)</vul>

    ipList = []
    port = ''
    protocol = ''
    hostName = ''

    <vul/>rules = answer.fetchall()</vul>
    if len(rules):
        print "[INFO] " + name + " " + str(len(rules)) + " IPs for domain " + domain + " in the database"
    if rules:
        for rule in rules:
            hostName = rule[1]
            ipList.append(rule[3])
            port = rule[4]
            protocol = rule[5]
        
        newIpList = get_dest_ip(domain)

        if set(ipList) == set(newIpList):
            print "[INFO] IPs for domain {0} stay the same, do not change iptables".format(domain)
        else:
            print "[INFO] Start Updating Rules for domain {0}".format(domain)
            update_iptable(ipList, newIpList, str(port), str(protocol).upper(), domain, name, hostName)

    conn.close()

def update_iptable(ipList, newIpList, port, protocol, domain, mac_addr, hostName):
    # clear databse for the domain
    conn = sqlite3.connect('device.db')
    cursor = conn.cursor()
    <vul/>old_query = "DELETE FROM DEVICE WHERE NAME = '{0}' AND DOMAIN = '{1}'".format(mac_addr, domain)
    cursor.execute(old_query)</vul>
    conn.commit()

    # remove old IP from iptables
    for oldIp in ipList:
        call('iptables -D FORWARD -p ' + protocol + ' -d '+ oldIp + ' --dport ' + port + ' -m mac --mac-source ' + mac_addr + ' -j ACCEPT', shell=True)
        print "[INFO] Remove old IP {0} from iptables".format(oldIp)

    # Append new
    for newIp in newIpList:
        # iptables
        call('iptables -I FORWARD -p ' + protocol + ' -d '+ newIp + ' --dport ' + port + ' -m mac --mac-source ' + mac_addr + ' -j ACCEPT', shell=True)
        print "[INFO] Add new IP {0} to iptables".format(newIp)
        # database
        <vul/>query = "INSERT INTO DEVICE(NAME, HOSTNAME, DOMAIN, IP, PORT, PROTOCOL) VALUES('{0}','{1}','{2}','{3}','{4}', '{5}')".format(mac_addr, hostName, domain, newIp, port, protocol)
        cursor.execute(query)</vul>
        conn.commit()
        print "[INFO] Add new IP {0} to database".format(newIp)
    
    conn.close()

def get_domain_list(pkt):
    try:
        conn = sqlite3.connect('device.db')
    except:
        print "[ERROR] Fail to connect to database"
        return []
    
    cursor = conn.cursor()

    query = "SELECT DOMAIN FROM DEVICE WHERE NAME = '{0}'".format(mac_addr)
    answer = cursor.execute(query)

    res = []
    <vul/>for domain in answer.fetchall():</vul>
        res.append(domain)
    
    return list(set(res))
