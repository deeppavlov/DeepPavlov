AWS DeepPavlov ODQA deployment manual
=====================================

Here is a manual for deployment DeepPavlov ODQA in Amazon Web Services using EC2 virtual machine.

Deployment process consists of two main stages:

1. AWS EC2 machine launch
2. DeepPavlov ODQA deployment

1. AWS EC2 machine launch
-------------------------

1.  Login to your AWS console and proceed to the EC2 services dashboard.

.. image:: ../_static/aws_ec2/01_login_to_aws.png
   :width: 800

2.  Choose Ubuntu Server 18.04 LTS 64-bit x86 machine.

.. image:: ../_static/aws_ec2/02_choose_ubuntu.png
   :width: 800

3.  You should select appropriate instance type because of high memory consumption by ODQA.
    32 GiB memory is a minimum. Then press *"Next: ..."*

.. image:: ../_static/aws_ec2/03_select_instance_type.png
   :width: 800

4.  Proceed to Step 4. Your instance storage size should be no less than 50 GiB to
    store ODQA components.

.. image:: ../_static/aws_ec2/04_add_storage.png
   :width: 800

5.  Proceed to Step 7. Check your instance parameters and press *"Launch"* button.
    You will be prompted to create and save security key pair for further access to your instance.

.. image:: ../_static/aws_ec2/05_review_instance.png
   :width: 800

6.  Return to your EC2 services dashboard and navigate to your running instances list.

.. image:: ../_static/aws_ec2/06_go_to_running_instances.png
   :width: 800

7.  Wait until instance initializing finishes (instance status become *"running"*).

.. image:: ../_static/aws_ec2/07_wait_init.png
   :width: 800

8.  To make DeepPavlov ODQA model rest API accessible from Internet you should set
    corresponding inbound security rules:

    8.1 Navigate to your instance security group dashboard
    (in this example security group has name *"launch-wizard-2"*).

    .. image:: ../_static/aws_ec2/08_01_set_sec_group.png
       :width: 800

    8.2 Select *"Inbound"* rules tab, click *"Edit"*, then click *"Add Rule"*.
    For your new rule select *"Custom TCP Rule"* type, *"Anywhere"* source and input
    port for your ODQA API. Click *"Save"*.

    .. image:: ../_static/aws_ec2/08_02_set_inbound.png
       :width: 800

9.  Connecting to your instance by SSH:

    9.1 Navigate to your instance dashboard, right-click your instance, select *"Connect"*.

    .. image:: ../_static/aws_ec2/09_01_select_connect.png
       :width: 800

    You will be redirected to connection instructions screen for your dashboard.
    Follow instructions for standalone SSH client. SSH connection bash command example will
    already contain valid user and host name. To connect to your Amazon instance just run
    the example with valid path to your saved key pair (instead of *"dp_key_pair.pem"*
    in this example).

    .. image:: ../_static/aws_ec2/09_02_connection_info.png
       :width: 800

2. DeepPavlov PDQA deployment
-----------------------------



