from __future__ import print_function
import httplib2
import os
import sys

from apiclient import discovery, errors
from oauth2client import client
from oauth2client import tools
from oauth2client.file import Storage

try:
    import argparse
    flags = argparse.ArgumentParser(parents=[tools.argparser]).parse_args()
except ImportError:
    flags = None

# If modifying these scopes, delete your previously saved credentials
# at ~/.credentials/drive-python-quickstart.json
SCOPES = 'https://www.googleapis.com/auth/drive'
CLIENT_SECRET_FILE = 'client_secret.json'
APPLICATION_NAME = 'Drive Migration Tool'

PATH_ROOT = 'D:'

class Drive():
    """ Google Drive representation class
    """
    def __init__(self, name, service):
        self.name = name
        self.folders = set()
        self.root = None
        self.files = set()
        self.users = set()
        self.service = service

        # Initialise the drive
        self.build_drive()
        print ("Generating paths for <{0}>.".format(self.name))
        self.generate_paths() # This is so expensive, needs optimisation
        print ("Finished generating paths for <{0}>.".format(self.name))

    def add_folder(self, folder):
        """ Add a folder to the drive
        """
        self.folders.add(folder)

    def add_file(self, file):
        """ Add a file to the drive
        """
        self.files.add(file)

    def parse_path(self, path):
        """ Parse a given file path, returning an ordered list of path objects
        """
        if PATH_ROOT not in path:
            print ("Invalid path <{0}>.".format(path))
            return None

        if path == PATH_ROOT:
            # We're looking at root
            return [self.root]

        # Strip out the prefix
        path = path.strip(PATH_ROOT)
        path_list = path.split('/')

        # Build the list of path objects
        path_objects = []
        curr_item = self.root
        for item in path_list:
            # Check the folders for matches
            for folder in self.folders:
                if curr_item.id in folder.parents and folder.name == item:
                    # Add the object to the list
                    # Set the current folder
                    curr_item = folder

                    # Add it to the list
                    path_objects.append(curr_item)

                    # Hope there aren't duplicates
                    break;

            # Check the files for matches
            for file in self.files:
                if curr_item.id in file.parents and file.name == item:
                    # Set the current file
                    curr_item = file

                    # Add the object to the list
                    path_objects.append(curr_item)

                    # Hope there aren't duplicates
                    break;

        if not any(obj.name == path_list[-1] for obj in path_objects) or len(path_objects) == 0:
            print ("Path <{0}> could not be found. Only found: <{1}>".format(PATH_ROOT+path, path_objects))
            for obj in path_objects:
                print (obj)
            return None

        # Return the path objects
        return path_objects

    def generate_paths(self, curr_item=None, curr_path=None):
        """ Generate all the paths for every file and folder in the drive - expensive
        """
        # Start in root
        if not curr_item:
            curr_item = self.root

        # Build folders first
        if not curr_path:
            curr_path = []

        for file in self.files:
            path_objects = list(curr_path)
            if curr_path:
                curr_item = path_objects[-1]

            if not file.path and curr_item.id in file.parents:
                path_string = curr_item.path + "/" + file.name
                # Build the path
                # path_objects.append(file)
                # path_string = self.build_path_string(path_objects)

                # Set the file path string
                file.set_path(path_string)

        for folder in self.folders:
            # Reset to the current path
            # print (curr_path)
            path_objects = list(curr_path)
            if curr_path:
                curr_item = path_objects[-1]

            if not folder.path and curr_item.id in folder.parents:
                path_string = curr_item.path + "/" + folder.name
                # Build the path
                # path_objects.append(folder)
                # path_string = self.build_path_string(path_objects)

                # Set the folder path string
                folder.set_path(path_string)

                # Generate children
                self.generate_paths(folder, path_objects)


    def build_path_string(self, path_objects):
        """ Build the path string from a given list of path_objects
        """ 
        path_string = PATH_ROOT
        end_index = len(path_objects) - 1

        for index, obj in enumerate(path_objects):
            if index == end_index:
                path_string = path_string + obj.name
            else:
                path_string = path_string + obj.name + "/"

        return path_string

    def build_path(self, file, return_string=False):
        """ Build the path to a given file
        """
        curr_item = file
        path_objects = []
        while curr_item != self.root:
            for folder in self.folders:
                if folder.id in curr_item.parents:
                    path_objects.insert(0, curr_item)
                    curr_item = folder

            print ("Path to <{0}> could not be built.".format(file.name))
            break;

        if return_string:
            self.build_path_string(path_objects)
        else:
            return path_objects

    def get_folder(self, name=None, id=None, parent=None):
        """ Get a folder by name or id
        """
        for folder in self.folders:
            if parent:
                parent_folder = self.get_folder(name=parent)
                if (folder.id == id or folder.name == name) and parent_folder.id in folder.parents:
                    return folder
            else:
                if folder.id == id or folder.name == name:
                    return folder

        print ("Could not find the folder with name: <{0}>, id: <{1}>, and parent: <{2}>. Aborting.".format(name, id, parent))
        return None

    def get_folders(self, name=None, id=None, parent_name=None):
        """ Get a list of folders which have a certain name
        """
        folder_set = set()
        for folder in self.folders:
            if parent_name:
                parent_folder = self.get_folder(name=parent_name)
                if (folder.id == id or folder.name == name) and parent_folder.id in folder.parents:
                    folder_set.add(folder)
            else:
                if folder.id == id or folder.name == name:
                    folder_set.add(folder)

        if len(folder_set) > 0:
            return folder_set
        else:
            print ("Could not find a folder with name: <{0}>, id: <{1}>, and parent: <{2}>. Aborting.".format(name, id, parent_name))
            sys.exit()

    def get_file(self, name=None, id=None, parent_name=None):
        """ Get a file by name or id
        """
        for file in self.files:
            if parent_name:
                possible_folders = self.get_folders(name=parent_name)
                for parent_folder in possible_folders:
                    if (file.id == id or file.name == name) and parent_folder.id in file.parents:
                        return file
            else:
                if file.id == id or file.name == name:
                    return file

        print ("Could not find the file with name: <{0}>, id: <{1}>, and parent: <{2}> in <{3}>.".format(name, id, parent_name, self.name))
        return None

    def print_drive(self, base_folder_path=PATH_ROOT, verbose=False, curr_folder=None, prefix=""):
        """ Print the drive structure from the given root folder
        """
        # If we're looking at the base folder, set it as the current folder
        if not curr_folder:
            path_objects = self.parse_path(base_folder_path)
            if not path_objects:
                # Will already have thrown error message
                sys.exit()
            else:
                curr_folder = path_objects[-1]

        # Print the current folder
        if curr_folder is self.root:
            print (prefix + curr_folder.id, curr_folder.name.encode('utf-8') + " ("+curr_folder.owner.name.encode('utf-8')+")")
        elif verbose:
            print (prefix + curr_folder.id, curr_folder.name.encode('utf-8') + " ("+curr_folder.owner.name.encode('utf-8')+")" + " ("+curr_folder.last_modified_time.encode('utf-8')+")" + " ("+curr_folder.last_modified_by.email.encode('utf-8')+")")
        else:
            print (prefix + curr_folder.name.encode('utf-8'))

        # Print file(s)
        for file in self.files:
            if curr_folder.id in file.parents:
                if verbose:
                    print (prefix + "\t" + file.id, file.name.encode('utf-8') + " ("+file.owner.name.encode('utf-8')+")" + " ("+file.last_modified_time.encode('utf-8')+")" + " ("+file.last_modified_by.email.encode('utf-8')+")")
                else:
                    print (prefix + "\t" + file.name.encode('utf-8'))

        # Print child folder(s)
        for folder in self.folders:
            if folder.parents:
                if curr_folder.id in folder.parents:
                    self.print_drive(verbose=verbose, curr_folder=folder, prefix=prefix+"\t")

    def get_user_emails(self):
        """ Get all user emails
        """
        emails = set()
        for user in self.users:
            emails.add(user.email)
        return emails

    def add_user(self, user):
        """ Add a user to the drive
        """
        # Only add the user if they don't already exist
        if user.email not in self.get_user_emails():
            self.users.add(user)

    def update_drive(self, src_drive, base_folder_path=None, parent=None, curr_folder=None):
        """ Update the owner and last known modified date for a file/folder
        """
        # If we don't have a current folder, get one
        if not curr_folder:
            if not base_folder_path:
                curr_folder = self.root
            else:
                curr_folder = src_drive.get_folder_via_path(base_folder_path)

        # Update files in current folder
        file_count = 0
        for file in src_drive.files:
            if curr_folder.id in file.parents:
                self.update_info(src_drive=src_drive, path=file.path)
                file_count += 1

        print ("Updated <{0}> files in folder <{1}> in <{2}> drive.".format(file_count, curr_folder.name, self.name))

        # Update sub-folders
        for folder in src_drive.folders:
            if curr_folder.id in folder.parents:
                <vul/>result = self.update_drive(src_drive=src_drive, parent=curr_folder, curr_folder=folder)
                print (result)</vul>

    def get_file_via_path(self, path):
        """ Get a file via its path
        """
        for file in self.files:
            if file.path == path:
                return file

        <vul/>print ("Could not find file at <{0}> in <{1}>.".format(path))</vul>

    def get_folder_via_path(self, path):
        """ Get a folder via its path
        """
        for folder in self.folders:
            if folder.path == path:
                return folder

        <vul/>print ("Could not find folder at <{0}> in <{1}>.".format(path))</vul>

    <vul/>def update_info(self, src_drive, path, update_owner=False):</vul>
        """ Update the supplied drive item with the last known modified date and owner for a given file
        """
        <vul/>src_item = src_drive.get_file_via_path(path)
        dest_item = self.get_file_via_path(path)</vul>

        # Make sure both the source and destination files can be found
        if not src_item:
            return "Item could not be found in <{0}>. Skipping.".format(src_drive.name)
        if not dest_item:
            return "Item could not be found in <{0}>. Skipping.".format(self.name)

        try:
            # Build the updated payload
            time_body = {'modifiedTime': src_item.last_modified_time}

            # Send the file back
            time_response = self.service.files().update(fileId=dest_item.id, 
                                                        body=time_body
                                                        ).execute()

            if update_owner:
                # Get the new user email
                new_user = convert_to_new_domain(src_item.last_modified_by.email, NEW_DOMAIN)

                # Build the updated payload
                user_body = {'emailAddress': new_user,
                             'role': 'owner',
                             'type': 'user'}

                # Send the file back
                user_response = self.service.permissions().create(fileId=dest_item.id, 
                                                                  body=user_body,
                                                                  sendNotificationEmail=True,
                                                                  transferOwnership=True
                                                                  ).execute()

            return "Successfully updated <{0}> in drive <{1}>.".format(dest_item.name.encode('utf-8'), self.name)

        except errors.HttpError, error:
            print ("An error occurred: {0}".format(error))
            sys.exit()

    def build_drive(self):
        """ Build the Google drive for the given service
        """
        page_token = None
        page_no = 1
        print ("Retrieving drive data for <{0}>...".format(self.name))

        # Get the root "My Drive" folder first
        response = self.service.files().get(fileId='root',
                                            fields="id, mimeType, name, owners").execute()
        owner = User(name=response['owners'][0]['displayName'], email=response['owners'][0]['emailAddress'])
        root_folder = Folder(id=response['id'], 
                        name=response['name'], 
                        owner=owner, 
                        parents=None,
                        last_modified_time=None,
                        last_modified_by=None,
                        path=PATH_ROOT)
        self.root = root_folder

        # Get the rest of the drive
        while True:
            # Get entire folder structure, we'll work down from here
            response = self.service.files().list(q="trashed = false",
                                            pageSize=1000,
                                            pageToken=page_token,
                                            fields="nextPageToken, files(id, mimeType, name, owners, parents, modifiedTime, lastModifyingUser)").execute()
            results = response.get('files', [])

            # Build folder structure in memory
            for result in results:
                # Create owner
                owner = User(name=result['owners'][0]['displayName'], email=result['owners'][0]['emailAddress'])
                self.add_user(owner)

                # Save last modifying user, if it exists
                if 'emailAddress' in result['lastModifyingUser']:
                    modified_by = User(name=result['lastModifyingUser']['displayName'], email=result['lastModifyingUser']['emailAddress'])
                    self.add_user(modified_by)
                else:
                    modified_by = None

                # Check if it's a root folder
                if 'parents' in result:
                    parents = result['parents']
                else:
                    parents = 'root'

                # Create drive item
                if result['mimeType'] == 'application/vnd.google-apps.folder':
                    folder = Folder(id=result['id'], 
                                    name=result['name'], 
                                    owner=owner, 
                                    parents=parents,
                                    last_modified_time=result['modifiedTime'],
                                    last_modified_by=modified_by)
                    self.add_folder(folder)
                else:
                    file = File(id=result['id'], 
                                name=result['name'], 
                                owner=owner, 
                                parents=parents,
                                last_modified_time=result['modifiedTime'],
                                last_modified_by=modified_by,
                                mime_type=result['mimeType'])
                    self.add_file(file)

            # Look for more pages of results
            page_token = response.get('nextPageToken', None)
            page_no += 1
            if page_token is None:
                break;

        print ("Found <{0}> pages of results for <{1}>. Drive has been built.".format(page_no, self.name))

class User():
    """ User representation class
    """
    def __init__(self, name, email):
        self.name = name
        self.email = email

    def __repr__(self):
        return "<user: {0}>".format(self.email)

class File():
    """ File representation class
    """
    def __init__(self, id, name, owner, parents, last_modified_time, last_modified_by, mime_type):
        self.id = id
        self.name = name
        self.parents = parents
        self.owner = owner
        self.last_modified_time = last_modified_time
        self.last_modified_by = last_modified_by
        self.mime_type = mime_type

        self.path = None

    def __repr__(self):
        return "<file: {0}>".format(self.name)

    def set_path(self, path):
        """ Set the file path string
        """
        self.path = path

class Folder():
    """ Folder representation class
    """
    def __init__(self, id, name, owner, parents=None, last_modified_time=None, last_modified_by=None, path=None):
        self.id = id
        self.name = name
        self.parents = parents
        self.owner = owner
        self.last_modified_time = last_modified_time
        self.last_modified_by = last_modified_by

        self.path = path

    def __repr__(self):
        return "<folder: {0}>".format(self.name)

    def set_path(self, path):
        """ Set the folder path string
        """
        self.path = path

def convert_to_new_domain(email, new_domain):
    """ Convert a given email to a new domain
    """
    return email.split('@')[0].encode('utf-8')+'@'+new_domain

def get_credentials(src):
    """Gets valid user credentials from storage.

    If nothing has been stored, or if the stored credentials are invalid,
    the OAuth2 flow is completed to obtain the new credentials.

    Returns:
        Credentials, the obtained credential.
    """
    home_dir = os.path.expanduser('~')
    credential_dir = os.path.join(home_dir, '.credentials')
    if not os.path.exists(credential_dir):
        os.makedirs(credential_dir)
    if src == 'src':
        credential_path = os.path.join(credential_dir,
                                   'src-drive-migration-tool.json')
    else:
        credential_path = os.path.join(credential_dir,
                                   'dest-drive-migration-tool.json')

    store = Storage(credential_path)
    credentials = store.get()
    if not credentials or credentials.invalid:
        flow = client.flow_from_clientsecrets(CLIENT_SECRET_FILE, SCOPES)
        flow.user_agent = APPLICATION_NAME
        if flags:
            credentials = tools.run_flow(flow, store, flags)
        else: # Needed only for compatibility with Python 2.6
            credentials = tools.run(flow, store)
        print('Storing credentials to ' + credential_path)

    return credentials

def main():
    # Source account credentials
    src_credentials = get_credentials('src')
    src_http = src_credentials.authorize(httplib2.Http())
    src_service = discovery.build('drive', 'v3', http=src_http)

    # Destination account credentials
    dest_credentials = get_credentials('dest')
    dest_http = dest_credentials.authorize(httplib2.Http())
    dest_service = discovery.build('drive', 'v3', http=dest_http)

    src_drive = Drive("source drive", src_service)
    dest_drive = Drive("destination drive", dest_service)

    # Print the drive structure

    # Test updating file
    # parent = 'nov-16'
    # dest_drive.update_info(src_drive, fname, parent)

    # for folder in src_drive.folders:
    #     if folder.name == '2016-06':
    #         print (folder.id, folder.name, folder.path)

    # Test full drive update


if __name__ == '__main__':
    main()
