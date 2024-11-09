#!/usr/bin/env python3

import argparse
import logging
import os
import shutil

from .pyweb import (# DEFAULT_PYWEB_HOME,
                    # DEFAULT_PYWEB_VAR,
                    DEFAULT_PYWEB_CONTENT_DIR,
                    DEFAULT_PYWEB_LOG_DIR)


class ContentInstallerException(Exception):
    pass

class ContentInstaller:
    def __init__(self, src_path, dst_path, www_root, logger=None):
        self.logger = logger or logging.getLogger("ContentInstaller")
        _src, _dst, _www = self._sanity_check_path(src_path, dst_path, www_root)
        self.src_path = _src
        self.dst_path = _dst
        self.www_root = _www


    def _sanity_check_path(self, src, dst, www_root):
        logger = self.logger
        if not os.path.isdir(src):
            msg = "Source path: %s does not exist or is not a directory." % src
            logger.critical(msg)
            raise ContentInstallerException(msg)

        if not os.path.isdir(www_root):
            msg = "Web root % s does not exist or is not a directory." % src
            logger.critical(msg)
            raise ContentInstallerException(msg)

        www_root_abs = os.path.abspath(www_root)

        rel_dst = dst
        if os.path.isabs(dst):
            <fix/>_dst = os.path.realpath(dst)
            _root = os.path.commonprefix([www_root_abs, _dst])</fix>
            if _root is not www_root_abs:
                msg = "Destination path is absolute and is not a subdirectory of web root. {}".format([www_root, dst])
                logger.critical(msg)
                raise ContentInstallerException(msg)

            rel_dst = os.path.relpath(_dst, www_root_abs)
        else:
            _dst = os.path.join(www_root_abs, dst)
            _dst = os.path.realpath(_dst)
            _root = os.path.commonprefix([www_root_abs, _dst])
            if _root is not www_root_abs:
                msg = "Destination is a relative path that resolves outside of web root. {}".format([www_root_abs, dst])
                logger.critical(msg)
                raise ContentInstallerException(msg)

            rel_dst = os.path.relpath(_dst, www_root_abs)

        abs_dst = os.path.join(www_root_abs, rel_dst)
        if os.path.exists(abs_dst):
            msg = "Destination directory already exists: {}".format(abs_dst)
            logger.critical(msg)
            raise ContentInstallerException(msg)

        return (src, rel_dst, www_root_abs)

    def install(self):
        logger = self.logger
        src_path = self.src_path
        dst_path = os.path.join(self.www_root, self.dst_path)
        logger.info("Copying %s to %s" % (src_path, dst_path))
        try:
            shutil.copytree(src_path, dst_path, symlinks=True)
        except Exception as e:
            logger.critical("Exception: {}".format(e))
            raise


def parse_args(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument("content_src_dir", help="Source directory containing content to be intalled.")
    parser.add_argument("content_dst_dir", help="Name of directory under <WWW-ROOT> for content to be located.")
    parser.add_argument("--www-root", help="WWW root path to install to.")
    parser.add_argument("--log-path", help="Directory to write logfiles to.")
    args = parser.parse_args(argv)

    return args


def main(argv):
    args = parse_args(argv)
    src_dir = args.content_src_dir
    dst_dir = args.content_dst_dir

    www_root = args.www_root or DEFAULT_PYWEB_CONTENT_DIR
    logpath = args.log_path or DEFAULT_PYWEB_LOG_DIR

    logging.basicConfig(filename=os.path.join(logpath, "pyweb-installer.log"), level=logging.DEBUG)

    logger = logging.getLogger("pyweb-installer")
    print("Installing %s" % src_dir)
    print("Logging to %s" % os.path.join(logpath, "pyweb-installer.log"))
    try:
        installer = ContentInstaller(src_dir, dst_dir, www_root, logger=logger)
        installer.install()
    except Exception as e:
        print("Installation failed: %s" % str(e))
        logger.critical("Installation failed.")
        exit(1)

    print("Installation complete.")
    logger.info("Installation complete.")
