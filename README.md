PDF Converstional ChatBot

admin - folder has all application code for admin application.
user - folder has all application code for user application.

Make sure you are connected to AWS also, since you need to do some operation in S3. You can mount ~/.aws/crdentials to docker volume instead of passing it as environment variables.
