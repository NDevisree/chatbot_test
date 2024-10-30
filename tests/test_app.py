import unittest
from io import BytesIO
from app import app, get_Chat_response

class FlaskAppTests(unittest.TestCase):

    def setUp(self):
        # Set up the test client and context
        self.app = app.test_client()
        self.app.testing = True

    def test_index(self):
        # Test the index route
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Chat', response.data)  # Adjust based on your HTML content

    def test_chat_response(self):
        # Test the chat response function
        response = get_Chat_response("Hello")
        self.assertEqual(response, 'Yes')  # Adjust based on actual logic

    def test_chat_endpoint(self):
        # Test the chat endpoint with a sample message
        response = self.app.post('/get', data={'msg': 'Hello'})
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Assistant:', response.data)  # Check for expected output

    def test_file_upload(self):
        # Simulate file upload
        data = {
            'files[]': (BytesIO(b"dummy data"), 'test.pdf'),
        }
        response = self.app.post('/upload', data=data, content_type='multipart/form-data')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'success', response.data)  # Check for the success message

if __name__ == '__main__':
    unittest.main()
